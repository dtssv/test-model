"""
数据采集器基类
定义所有采集器的标准接口和通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Dict, Any, Optional
from enum import Enum
import hashlib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    IMAGE_TEXT = "image_text"
    AUDIO = "audio"
    VIDEO = "video"


class DataSourceType(Enum):
    """数据源类型枚举"""
    COMMON_CRAWL = "common_crawl"
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    GITHUB = "github"
    LAION = "laion"
    WEBVID = "webvid"
    LIBRISPEECH = "librispeech"
    CUSTOM = "custom"


@dataclass
class CollectionConfig:
    """数据采集通用配置"""
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 30
    max_workers: int = 10
    enable_dedup: bool = True
    quality_threshold: float = 0.5
    storage_bucket: str = "raw-data"
    metadata_db: str = "postgresql://localhost:5432/mllm_db"
    redis_url: str = "redis://localhost:6379"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    enable_progress_report: bool = True


@dataclass
class DataSource:
    """数据源配置"""
    source_type: DataSourceType
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_type': self.source_type.value,
            'source_url': self.source_url,
            'source_path': self.source_path,
            'metadata': self.metadata,
            'filters': self.filters
        }


@dataclass
class RawDataItem:
    """原始数据项"""
    data_id: str
    data_type: DataType
    content: Any  # 文本内容、图像bytes等
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[DataSource] = None
    collected_at: datetime = field(default_factory=datetime.now)
    
    def compute_hash(self) -> str:
        """计算数据项的唯一哈希值"""
        if isinstance(self.content, bytes):
            return hashlib.sha256(self.content).hexdigest()
        elif isinstance(self.content, str):
            return hashlib.sha256(self.content.encode('utf-8')).hexdigest()
        else:
            return hashlib.sha256(str(self.content).encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_id': self.data_id,
            'data_type': self.data_type.value,
            'metadata': self.metadata,
            'source': self.source.to_dict() if self.source else None,
            'collected_at': self.collected_at.isoformat()
        }


@dataclass
class BatchSaveResult:
    """批量保存结果"""
    total_items: int
    saved_items: int
    failed_items: int
    saved_ids: List[str]
    failed_ids: List[str]
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.saved_items / self.total_items


class BaseCollector(ABC):
    """
    所有数据采集器的抽象基类。
    定义采集、存储、上报进度的标准接口。
    """
    
    def __init__(
        self,
        config: CollectionConfig,
        storage_client=None,
        metadata_store=None
    ):
        """
        初始化采集器
        
        Args:
            config: 采集配置
            storage_client: MinIO存储客户端
            metadata_store: PostgreSQL元数据存储
        """
        self.config = config
        self.storage_client = storage_client
        self.metadata_store = metadata_store
        self.collected_count = 0
        self.error_count = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def collect(self, source: DataSource) -> AsyncIterator[RawDataItem]:
        """
        从指定数据源异步采集原始数据
        
        Args:
            source: 数据源配置
            
        Yields:
            RawDataItem: 采集到的原始数据项
        """
        pass
    
    async def save_batch(self, items: List[RawDataItem]) -> BatchSaveResult:
        """
        批量保存到MinIO，同时写入元数据
        
        Args:
            items: 待保存的数据项列表
            
        Returns:
            BatchSaveResult: 保存结果
        """
        saved_ids = []
        failed_ids = []
        errors = []
        
        for item in items:
            try:
                # 保存到MinIO
                object_name = f"{item.data_type.value}/{item.data_id}"
                if isinstance(item.content, bytes):
                    await self.storage_client.upload_bytes(
                        bucket=self.config.storage_bucket,
                        object_name=object_name,
                        data=item.content
                    )
                elif isinstance(item.content, str):
                    await self.storage_client.upload_bytes(
                        bucket=self.config.storage_bucket,
                        object_name=object_name,
                        data=item.content.encode('utf-8')
                    )
                
                # 写入元数据
                if self.metadata_store:
                    await self.metadata_store.insert_item({
                        'data_id': item.data_id,
                        'data_type': item.data_type.value,
                        'object_path': object_name,
                        'metadata': item.metadata,
                        'source': item.source.to_dict() if item.source else None,
                        'collected_at': item.collected_at,
                        'content_hash': item.compute_hash()
                    })
                
                saved_ids.append(item.data_id)
                
            except Exception as e:
                self.logger.error(f"Failed to save item {item.data_id}: {e}")
                failed_ids.append(item.data_id)
                errors.append(str(e))
        
        return BatchSaveResult(
            total_items=len(items),
            saved_items=len(saved_ids),
            failed_items=len(failed_ids),
            saved_ids=saved_ids,
            failed_ids=failed_ids,
            errors=errors
        )
    
    def report_progress(self, collected: int, total: int, errors: int) -> None:
        """
        上报采集进度到监控系统
        
        Args:
            collected: 已采集数量
            total: 总数量
            errors: 错误数量
        """
        self.collected_count = collected
        self.error_count = errors
        
        progress_pct = (collected / total * 100) if total > 0 else 0
        self.logger.info(
            f"Collection progress: {collected}/{total} ({progress_pct:.2f}%), "
            f"errors: {errors}"
        )
        
        # TODO: 上报到Prometheus或其他监控系统
        # 可以通过Redis或消息队列发送进度信息
    
    def validate_item(self, item: RawDataItem) -> bool:
        """
        基础数据校验(格式、大小、完整性)
        
        Args:
            item: 待校验的数据项
            
        Returns:
            bool: 是否通过校验
        """
        # 检查数据ID
        if not item.data_id or not isinstance(item.data_id, str):
            self.logger.warning(f"Invalid data_id: {item.data_id}")
            return False
        
        # 检查数据内容
        if item.content is None:
            self.logger.warning(f"Empty content for item {item.data_id}")
            return False
        
        # 检查数据类型
        if not isinstance(item.data_type, DataType):
            self.logger.warning(f"Invalid data_type: {item.data_type}")
            return False
        
        # 根据数据类型进行特定校验
        if item.data_type == DataType.TEXT:
            if isinstance(item.content, str):
                # 检查文本长度
                if len(item.content.strip()) < 10:
                    self.logger.warning(f"Text too short for item {item.data_id}")
                    return False
            else:
                self.logger.warning(f"Invalid text content type for item {item.data_id}")
                return False
        
        elif item.data_type == DataType.IMAGE:
            if isinstance(item.content, bytes):
                # 检查图像大小 (最小1KB, 最大50MB)
                size_kb = len(item.content) / 1024
                if size_kb < 1 or size_kb > 50 * 1024:
                    self.logger.warning(f"Image size out of range for item {item.data_id}: {size_kb:.2f}KB")
                    return False
            else:
                self.logger.warning(f"Invalid image content type for item {item.data_id}")
                return False
        
        return True
    
    async def __aiter__(self):
        """支持异步迭代"""
        return self