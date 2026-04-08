"""
MinIO对象存储客户端封装
用于存储原始数据、清洗后数据、模型checkpoint等大文件
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from io import BytesIO
from datetime import timedelta
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ObjectInfo:
    """对象信息"""
    object_name: str
    bucket_name: str
    size: int
    etag: str
    last_modified: str
    content_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'object_name': self.object_name,
            'bucket_name': self.bucket_name,
            'size': self.size,
            'etag': self.etag,
            'last_modified': self.last_modified,
            'content_type': self.content_type
        }


class MinIOClient:
    """
    MinIO对象存储客户端封装。
    用于存储原始数据、清洗后数据、模型checkpoint等大文件。
    """
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: str = None
    ):
        """
        初始化MinIO客户端
        
        Args:
            endpoint: MinIO服务地址 (如: localhost:9000)
            access_key: 访问密钥
            secret_key: 私密密钥
            secure: 是否使用HTTPS
            region: 区域
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.region = region
        self.client = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._init_client()
    
    def _init_client(self):
        """初始化MinIO客户端"""
        try:
            from minio import Minio
            from minio.error import S3Error
            
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region
            )
            self.logger.info(f"MinIO client initialized: {self.endpoint}")
            
        except ImportError:
            self.logger.error("minio package not installed. Please install it: pip install minio")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    
    async def ensure_bucket(self, bucket: str) -> bool:
        """
        确保桶存在，不存在则创建
        
        Args:
            bucket: 桶名称
            
        Returns:
            bool: 是否成功
        """
        try:
            # MinIO客户端是同步的，需要在异步环境中运行
            loop = asyncio.get_event_loop()
            
            if not await loop.run_in_executor(None, self.client.bucket_exists, bucket):
                await loop.run_in_executor(None, self.client.make_bucket, bucket)
                self.logger.info(f"Created bucket: {bucket}")
            else:
                self.logger.debug(f"Bucket already exists: {bucket}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ensure bucket {bucket}: {e}")
            return False
    
    async def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str,
        metadata: Dict[str, str] = None
    ) -> str:
        """
        上传文件到MinIO
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            file_path: 本地文件路径
            metadata: 元数据
            
        Returns:
            str: 对象URL
        """
        try:
            # 确保桶存在
            await self.ensure_bucket(bucket)
            
            # 上传文件
            loop = asyncio.get_event_loop()
            
            import os
            file_size = os.path.getsize(file_path)
            content_type = self._get_content_type(file_path)
            
            await loop.run_in_executor(
                None,
                lambda: self.client.fput_object(
                    bucket,
                    object_name,
                    file_path,
                    metadata=metadata,
                    content_type=content_type
                )
            )
            
            url = f"http://{self.endpoint}/{bucket}/{object_name}"
            self.logger.info(f"Uploaded file: {url}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to upload file {file_path}: {e}")
            raise
    
    async def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        metadata: Dict[str, str] = None,
        content_type: str = None
    ) -> str:
        """
        上传字节数据到MinIO
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            data: 字节数据
            metadata: 元数据
            content_type: 内容类型
            
        Returns:
            str: 对象URL
        """
        try:
            # 确保桶存在
            await self.ensure_bucket(bucket)
            
            # 上传数据
            loop = asyncio.get_event_loop()
            
            if content_type is None:
                content_type = 'application/octet-stream'
            
            data_stream = BytesIO(data)
            data_size = len(data)
            
            await loop.run_in_executor(
                None,
                lambda: self.client.put_object(
                    bucket,
                    object_name,
                    data_stream,
                    data_size,
                    metadata=metadata,
                    content_type=content_type
                )
            )
            
            url = f"http://{self.endpoint}/{bucket}/{object_name}"
            self.logger.debug(f"Uploaded bytes: {url}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to upload bytes to {object_name}: {e}")
            raise
    
    async def download_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str
    ) -> bool:
        """
        从MinIO下载文件
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            file_path: 本地文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                lambda: self.client.fget_object(bucket, object_name, file_path)
            )
            
            self.logger.info(f"Downloaded file: {bucket}/{object_name} -> {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download file {object_name}: {e}")
            return False
    
    async def download_bytes(
        self,
        bucket: str,
        object_name: str
    ) -> Optional[bytes]:
        """
        从MinIO下载字节数据
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            
        Returns:
            Optional[bytes]: 字节数据，失败返回None
        """
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_object(bucket, object_name)
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to download bytes from {object_name}: {e}")
            return None
    
    async def list_objects(
        self,
        bucket: str,
        prefix: str = None,
        recursive: bool = True
    ) -> List[ObjectInfo]:
        """
        列出桶中指定前缀的对象
        
        Args:
            bucket: 桶名称
            prefix: 对象前缀
            recursive: 是否递归列出
            
        Returns:
            List[ObjectInfo]: 对象列表
        """
        try:
            loop = asyncio.get_event_loop()
            
            objects = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_objects(
                    bucket,
                    prefix=prefix,
                    recursive=recursive
                ))
            )
            
            result = []
            for obj in objects:
                info = ObjectInfo(
                    object_name=obj.object_name,
                    bucket_name=obj.bucket_name,
                    size=obj.size,
                    etag=obj.etag,
                    last_modified=obj.last_modified.isoformat() if obj.last_modified else '',
                    content_type=obj.content_type or ''
                )
                result.append(info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list objects in {bucket}: {e}")
            return []
    
    async def delete_object(
        self,
        bucket: str,
        object_name: str
    ) -> bool:
        """
        删除对象
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            
        Returns:
            bool: 是否成功
        """
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                lambda: self.client.remove_object(bucket, object_name)
            )
            
            self.logger.info(f"Deleted object: {bucket}/{object_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete object {object_name}: {e}")
            return False
    
    async def delete_objects(
        self,
        bucket: str,
        object_names: List[str]
    ) -> Dict[str, bool]:
        """
        批量删除对象
        
        Args:
            bucket: 桶名称
            object_names: 对象名称列表
            
        Returns:
            Dict[str, bool]: 删除结果
        """
        results = {}
        for object_name in object_names:
            results[object_name] = await self.delete_object(bucket, object_name)
        return results
    
    async def get_object_info(
        self,
        bucket: str,
        object_name: str
    ) -> Optional[ObjectInfo]:
        """
        获取对象信息
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            
        Returns:
            Optional[ObjectInfo]: 对象信息，不存在返回None
        """
        try:
            loop = asyncio.get_event_loop()
            
            stat = await loop.run_in_executor(
                None,
                lambda: self.client.stat_object(bucket, object_name)
            )
            
            return ObjectInfo(
                object_name=stat.object_name,
                bucket_name=stat.bucket_name,
                size=stat.size,
                etag=stat.etag,
                last_modified=stat.last_modified.isoformat() if stat.last_modified else '',
                content_type=stat.content_type or ''
            )
            
        except Exception as e:
            self.logger.debug(f"Object not found: {object_name}")
            return None
    
    async def generate_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires: int = 3600
    ) -> Optional[str]:
        """
        生成预签名URL
        
        Args:
            bucket: 桶名称
            object_name: 对象名称
            expires: 过期时间（秒）
            
        Returns:
            Optional[str]: 预签名URL
        """
        try:
            loop = asyncio.get_event_loop()
            
            url = await loop.run_in_executor(
                None,
                lambda: self.client.presigned_get_object(
                    bucket,
                    object_name,
                    expires=timedelta(seconds=expires)
                )
            )
            
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def _get_content_type(self, file_path: str) -> str:
        """
        根据文件扩展名获取内容类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 内容类型
        """
        import os
        _, ext = os.path.splitext(file_path)
        
        content_types = {
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.jsonl': 'application/jsonl',
            '.csv': 'text/csv',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.parquet': 'application/parquet',
            '.bin': 'application/octet-stream',
            '.pt': 'application/octet-stream',
            '.pth': 'application/octet-stream',
        }
        
        return content_types.get(ext.lower(), 'application/octet-stream')
    
    async def close(self):
        """关闭客户端连接"""
        # MinIO客户端不需要显式关闭
        self.logger.info("MinIO client closed")