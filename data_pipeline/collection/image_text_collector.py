"""
图文对数据采集器
支持从LAION-5B、CC3M/CC12M、DataComp等数据集采集图文对
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple
import asyncio
import aiohttp
import logging
from PIL import Image
import io
import hashlib
import uuid
from datetime import datetime
import json

from .base_collector import (
    BaseCollector, CollectionConfig, DataSource,
    RawDataItem, DataType, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class ImageTextCollectionConfig(CollectionConfig):
    """图文对采集专用配置"""
    min_image_size: int = 256  # 最小图像尺寸
    max_image_size: int = 8192  # 最大图像尺寸
    min_image_file_size: int = 1024  # 最小文件大小 (1KB)
    max_image_file_size: int = 50 * 1024 * 1024  # 最大文件大小 (50MB)
    allowed_formats: List[str] = field(default_factory=lambda: ['JPEG', 'PNG', 'WebP'])
    min_caption_length: int = 10  # 最小描述长度
    max_caption_length: int = 1000  # 最大描述长度
    # LAION配置
    laion_aesthetic_score_min: float = 0.0
    laion_aesthetic_score_max: float = 10.0
    # CLIP评分配置
    enable_clip_scoring: bool = True
    min_clip_score: float = 0.2
    # 下载配置
    download_timeout: int = 30
    max_concurrent_downloads: int = 100
    retry_attempts: int = 3


class ImageTextCollector(BaseCollector):
    """
    图文对数据采集器。
    数据来源：LAION-5B子集、CC3M/CC12M、DataComp、
             ShareGPT4V、ALLaVA等公开图文数据集。
    """
    
    def __init__(
        self,
        config: ImageTextCollectionConfig,
        storage_client=None,
        metadata_store=None,
        clip_model=None
    ):
        """
        初始化图文对采集器
        
        Args:
            config: 采集配置
            storage_client: MinIO存储客户端
            metadata_store: 元数据存储
            clip_model: CLIP模型用于评分
        """
        super().__init__(config, storage_client, metadata_store)
        self.image_text_config = config
        self.clip_model = clip_model
        self.session = None
        self.semaphore = None
    
    async def _init_session(self):
        """初始化HTTP会话和并发控制"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.image_text_config.max_concurrent_downloads,
                limit_per_host=20
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
            self.semaphore = asyncio.Semaphore(
                self.image_text_config.max_concurrent_downloads
            )
    
    async def close(self):
        """关闭资源"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def collect(self, source: DataSource) -> AsyncIterator[RawDataItem]:
        """
        从指定数据源采集图文对数据
        
        Args:
            source: 数据源配置
            
        Yields:
            RawDataItem: 图文对数据项
        """
        await self._init_session()
        
        if source.source_type == DataSourceType.LAION:
            async for item in self.collect_laion(
                source.metadata.get('subset', 'laion2B-en'),
                source.filters
            ):
                yield item
        
        elif source.source_type == DataSourceType.CUSTOM:
            # 自定义数据源
            if source.source_url:
                async for item in self._collect_from_url_list(source):
                    yield item
        
        else:
            self.logger.warning(f"Unsupported source type for image-text: {source.source_type}")
    
    async def collect_laion(
        self,
        subset: str = 'laion2B-en',
        filters: Dict[str, Any] = None
    ) -> AsyncIterator[RawDataItem]:
        """
        从LAION-5B下载图文对
        使用img2dataset工具批量下载
        
        Args:
            subset: LAION子集名称
            filters: 过滤条件
            
        Yields:
            RawDataItem: 图文对数据项
        """
        # LAION数据集的parquet文件URL
        # 实际实现需要使用img2dataset工具
        # 这里提供框架实现
        
        laion_url_template = "https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-{part:05d}.parquet"
        
        try:
            # 使用img2dataset下载
            # 实际使用时需要安装img2dataset包
            # from img2dataset import download
            
            self.logger.info(f"Collecting from LAION subset: {subset}")
            
            # 示例：从URL列表下载
            # 这里简化处理，实际需要读取parquet文件获取URL列表
            # 然后批量下载图像和对应的文本
            
            # 模拟下载流程
            url_list = filters.get('url_list', []) if filters else []
            
            for url_info in url_list:
                image_url = url_info.get('url')
                caption = url_info.get('caption', '')
                
                # 下载图像
                image_data = await self.download_image(image_url)
                
                if image_data:
                    # 验证图像
                    if self.verify_image(image_data):
                        # 可选：计算CLIP评分
                        clip_score = None
                        if self.image_text_config.enable_clip_scoring and self.clip_model:
                            try:
                                image = Image.open(io.BytesIO(image_data))
                                clip_score = self.compute_clip_score(image, caption)
                                
                                if clip_score < self.image_text_config.min_clip_score:
                                    self.logger.debug(f"Image filtered by CLIP score: {clip_score}")
                                    continue
                            except Exception as e:
                                self.logger.error(f"Error computing CLIP score: {e}")
                        
                        item = RawDataItem(
                            data_id=str(uuid.uuid4()),
                            data_type=DataType.IMAGE_TEXT,
                            content={
                                'image': image_data,
                                'caption': caption
                            },
                            metadata={
                                'image_url': image_url,
                                'caption_length': len(caption),
                                'clip_score': clip_score,
                                'subset': subset,
                                'source': 'laion'
                            },
                            source=DataSource(
                                source_type=DataSourceType.LAION,
                                source_url=image_url,
                                metadata={'subset': subset}
                            )
                        )
                        
                        if self.validate_item(item):
                            yield item
                            
        except Exception as e:
            self.logger.error(f"Error collecting from LAION: {e}")
            self.error_count += 1
    
    async def collect_datacomp(
        self,
        pool_size: str = 'medium'
    ) -> AsyncIterator[RawDataItem]:
        """
        从DataComp数据池获取高质量图文对
        
        Args:
            pool_size: 数据池大小 ('small', 'medium', 'large')
            
        Yields:
            RawDataItem: 图文对数据项
        """
        # DataComp数据集配置
        # 实际实现需要访问HuggingFace数据集
        pass
    
    async def collect_sharegpt4v(self) -> AsyncIterator[RawDataItem]:
        """
        下载ShareGPT4V详细图像描述数据集
        
        Yields:
            RawDataItem: 图文对数据项
        """
        # ShareGPT4V数据集URL
        # 实际实现需要访问HuggingFace数据集
        pass
    
    async def download_image(
        self,
        url: str,
        timeout: int = None
    ) -> Optional[bytes]:
        """
        异步下载单张图像，带重试和超时机制
        
        Args:
            url: 图像URL
            timeout: 超时时间（秒）
            
        Returns:
            Optional[bytes]: 图像数据，失败返回None
        """
        if timeout is None:
            timeout = self.image_text_config.download_timeout
        
        for attempt in range(self.image_text_config.retry_attempts):
            try:
                async with self.semaphore:
                    async with self.session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            # 检查Content-Type
                            content_type = response.headers.get('Content-Type', '')
                            if 'image' in content_type:
                                # 检查文件大小
                                content_length = response.headers.get('Content-Length')
                                if content_length:
                                    size = int(content_length)
                                    if size < self.image_text_config.min_image_file_size:
                                        self.logger.debug(f"Image too small: {size} bytes")
                                        return None
                                    if size > self.image_text_config.max_image_file_size:
                                        self.logger.debug(f"Image too large: {size} bytes")
                                        return None
                                
                                image_data = await response.read()
                                return image_data
                        else:
                            self.logger.debug(f"Failed to download {url}: HTTP {response.status}")
                            
            except asyncio.TimeoutError:
                self.logger.debug(f"Timeout downloading {url} (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"Error downloading {url} (attempt {attempt + 1}): {e}")
            
            # 重试前等待
            if attempt < self.image_text_config.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    def verify_image(self, image_data: bytes) -> bool:
        """
        验证图像完整性
        
        Args:
            image_data: 图像二进制数据
            
        Returns:
            bool: 是否有效
        """
        try:
            # 使用Pillow打开并验证图像
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            
            # 检查格式
            if image.format not in self.image_text_config.allowed_formats:
                self.logger.debug(f"Invalid image format: {image.format}")
                return False
            
            # 检查尺寸
            width, height = image.size
            if width < self.image_text_config.min_image_size or height < self.image_text_config.min_image_size:
                self.logger.debug(f"Image too small: {width}x{height}")
                return False
            
            if width > self.image_text_config.max_image_size or height > self.image_text_config.max_image_size:
                self.logger.debug(f"Image too large: {width}x{height}")
                return False
            
            # 重新打开以获取完整图像信息（verify后需要重新打开）
            image = Image.open(io.BytesIO(image_data))
            
            # 检查是否损坏
            image.load()
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Invalid image: {e}")
            return False
    
    def compute_clip_score(
        self,
        image: Image.Image,
        text: str
    ) -> float:
        """
        计算图文对的CLIP相似度分数
        
        Args:
            image: PIL图像对象
            text: 文本描述
            
        Returns:
            float: CLIP相似度分数 (0-1)
        """
        if self.clip_model is None:
            return 0.0
        
        try:
            import torch
            from transformers import CLIPProcessor
            
            # 处理图像和文本
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # 计算相似度
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = logits_per_image.softmax(dim=1)[0, 0].item()
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing CLIP score: {e}")
            return 0.0
    
    async def _collect_from_url_list(
        self,
        source: DataSource
    ) -> AsyncIterator[RawDataItem]:
        """
        从URL列表采集图文对
        
        Args:
            source: 数据源配置
            
        Yields:
            RawDataItem: 图文对数据项
        """
        # 从source中获取URL列表
        url_list_path = source.source_path
        
        if not url_list_path:
            return
        
        try:
            # 读取URL列表文件
            # 支持JSONL格式：每行一个JSON对象，包含url和caption字段
            with open(url_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        image_url = data.get('url')
                        caption = data.get('caption', '')
                        
                        if not image_url or not caption:
                            continue
                        
                        # 下载图像
                        image_data = await self.download_image(image_url)
                        
                        if image_data and self.verify_image(image_data):
                            # 计算CLIP评分
                            clip_score = None
                            if self.image_text_config.enable_clip_scoring and self.clip_model:
                                try:
                                    image = Image.open(io.BytesIO(image_data))
                                    clip_score = self.compute_clip_score(image, caption)
                                    
                                    if clip_score < self.image_text_config.min_clip_score:
                                        continue
                                except Exception as e:
                                    self.logger.error(f"Error computing CLIP score: {e}")
                                    clip_score = 0.0
                            
                            item = RawDataItem(
                                data_id=str(uuid.uuid4()),
                                data_type=DataType.IMAGE_TEXT,
                                content={
                                    'image': image_data,
                                    'caption': caption
                                },
                                metadata={
                                    'image_url': image_url,
                                    'caption_length': len(caption),
                                    'clip_score': clip_score,
                                    'source': 'custom'
                                },
                                source=source
                            )
                            
                            if self.validate_item(item):
                                yield item
                                
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing JSON line: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing item: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading URL list: {e}")
            self.error_count += 1
    
    def validate_item(self, item: RawDataItem) -> bool:
        """
        验证图文对数据项
        
        Args:
            item: 待验证的数据项
            
        Returns:
            bool: 是否通过验证
        """
        # 基础验证
        if not super().validate_item(item):
            return False
        
        # 检查内容结构
        if not isinstance(item.content, dict):
            self.logger.warning(f"Invalid content type for image-text item")
            return False
        
        # 检查图像数据
        image_data = item.content.get('image')
        if not image_data or not isinstance(image_data, bytes):
            self.logger.warning(f"Missing or invalid image data")
            return False
        
        # 检查caption
        caption = item.content.get('caption')
        if not caption or not isinstance(caption, str):
            self.logger.warning(f"Missing or invalid caption")
            return False
        
        # 检查caption长度
        caption_len = len(caption.strip())
        if caption_len < self.image_text_config.min_caption_length:
            self.logger.debug(f"Caption too short: {caption_len}")
            return False
        
        if caption_len > self.image_text_config.max_caption_length:
            self.logger.debug(f"Caption too long: {caption_len}")
            return False
        
        return True