"""
打标器基类
定义数据标注的标准接口和通用功能
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import asyncio
from datetime import datetime


class LabelType(Enum):
    """标签类型"""
    CAPTION = "caption"  # 图像描述
    QA_PAIR = "qa_pair"  # QA对
    CLASSIFICATION = "classification"  # 分类标签
    QUALITY_SCORE = "quality_score"  # 质量评分
    SAFETY_LABEL = "safety_label"  # 安全标签
    SENTIMENT = "sentiment"  # 情感标签
    TOPIC = "topic"  # 主题标签
    CUSTOM = "custom"  # 自定义标签


class LabelStatus(Enum):
    """标签状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass
class LabelConfig:
    """打标器配置基类"""
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 300
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    use_gpu: bool = True
    num_workers: int = 4


@dataclass
class Label:
    """标签数据结构"""
    label_type: LabelType
    label_value: Union[str, int, float, Dict[str, Any], List[Any]]
    confidence: float = 1.0
    source: str = "auto"  # auto, human, hybrid
    annotator: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'label_type': self.label_type.value,
            'label_value': self.label_value,
            'confidence': self.confidence,
            'source': self.source,
            'annotator': self.annotator,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }


@dataclass
class LabeledDataItem:
    """标注后的数据项"""
    item_id: str
    original_id: str
    data_type: str  # text, image, audio, video, multimodal
    content: Any
    labels: List[Label] = field(default_factory=list)
    status: LabelStatus = LabelStatus.PENDING
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_label(self, label: Label):
        """添加标签"""
        self.labels.append(label)
        if self.status == LabelStatus.PENDING:
            self.status = LabelStatus.COMPLETED
    
    def get_label(self, label_type: LabelType) -> Optional[Label]:
        """获取指定类型的标签"""
        for label in self.labels:
            if label.label_type == label_type:
                return label
        return None
    
    def get_all_labels(self, label_type: Optional[LabelType] = None) -> List[Label]:
        """获取所有标签"""
        if label_type:
            return [l for l in self.labels if l.label_type == label_type]
        return self.labels
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'item_id': self.item_id,
            'original_id': self.original_id,
            'data_type': self.data_type,
            'content': self.content,
            'labels': [label.to_dict() for label in self.labels],
            'status': self.status.value,
            'errors': self.errors,
            'metadata': self.metadata,
        }


@dataclass
class LabelingStats:
    """打标统计信息"""
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_labels: int = 0
    avg_confidence: float = 0.0
    processing_time: float = 0.0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    
    def update(self, item: LabeledDataItem):
        """更新统计"""
        self.total_items += 1
        if item.status == LabelStatus.COMPLETED:
            self.successful_items += 1
        elif item.status == LabelStatus.FAILED:
            self.failed_items += 1
        
        self.total_labels += len(item.labels)
        
        # 更新标签分布
        for label in item.labels:
            label_key = f"{label.label_type.value}"
            self.label_distribution[label_key] = self.label_distribution.get(label_key, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_items': self.total_items,
            'successful_items': self.successful_items,
            'failed_items': self.failed_items,
            'success_rate': self.successful_items / max(self.total_items, 1),
            'total_labels': self.total_labels,
            'avg_labels_per_item': self.total_labels / max(self.total_items, 1),
            'avg_confidence': self.avg_confidence,
            'processing_time': self.processing_time,
            'label_distribution': self.label_distribution,
        }


class BaseLabeler(ABC):
    """
    打标器抽象基类。
    定义所有打标器的标准接口。
    """
    
    def __init__(self, config: LabelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stats = LabelingStats()
        
        # 模型和处理器
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # 缓存
        self._cache: Dict[str, Label] = {}
    
    @abstractmethod
    def label(self, items: List[Any]) -> List[LabeledDataItem]:
        """
        执行标注。
        
        Args:
            items: 待标注的数据项
            
        Returns:
            List[LabeledDataItem]: 标注后的数据项
        """
        pass
    
    @abstractmethod
    def validate(self, item: Any) -> bool:
        """
        验证数据项是否适合标注。
        
        Args:
            item: 数据项
            
        Returns:
            bool: 是否适合标注
        """
        pass
    
    def label_batch(self, items: List[Any]) -> List[LabeledDataItem]:
        """
        批量标注（支持分批）。
        
        Args:
            items: 待标注的数据项
            
        Returns:
            List[LabeledDataItem]: 标注后的数据项
        """
        all_labeled = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            self.logger.info(f"Labeling batch {i // batch_size + 1}/{(len(items) + batch_size - 1) // batch_size}")
            
            try:
                labeled_batch = self.label(batch)
                all_labeled.extend(labeled_batch)
                
                # 更新统计
                for item in labeled_batch:
                    self.stats.update(item)
                
            except Exception as e:
                self.logger.error(f"Error labeling batch: {e}")
                # 创建失败项
                for item in batch:
                    labeled_item = LabeledDataItem(
                        item_id=f"labeled_{id(item)}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='unknown',
                        content=None,
                        status=LabelStatus.FAILED,
                        errors=[str(e)]
                    )
                    all_labeled.append(labeled_item)
                    self.stats.update(labeled_item)
        
        # 计算平均置信度
        if self.stats.total_labels > 0:
            total_confidence = sum(
                label.confidence
                for item in all_labeled
                for label in item.labels
            )
            self.stats.avg_confidence = total_confidence / self.stats.total_labels
        
        return all_labeled
    
    async def label_async(self, items: List[Any]) -> List[LabeledDataItem]:
        """
        异步标注。
        
        Args:
            items: 待标注的数据项
            
        Returns:
            List[LabeledDataItem]: 标注后的数据项
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.label_batch, items)
    
    def get_stats(self) -> LabelingStats:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = LabelingStats()
    
    def _get_cache_key(self, item: Any) -> Optional[str]:
        """
        生成缓存键。
        
        Args:
            item: 数据项
            
        Returns:
            Optional[str]: 缓存键
        """
        # 默认实现，子类可覆盖
        if hasattr(item, 'data_id'):
            return f"{item.data_id}_{hash(str(item))}"
        elif isinstance(item, dict) and 'data_id' in item:
            return f"{item['data_id']}_{hash(str(item))}"
        else:
            return None
    
    def _get_cached_label(self, key: str) -> Optional[Label]:
        """
        从缓存获取标签。
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Label]: 缓存的标签
        """
        if self.config.enable_cache and key in self._cache:
            return self._cache[key]
        return None
    
    def _cache_label(self, key: str, label: Label):
        """
        缓存标签。
        
        Args:
            key: 缓存键
            label: 标签
        """
        if self.config.enable_cache:
            self._cache[key] = label
    
    def _create_labeled_item(
        self,
        item: Any,
        labels: List[Label],
        status: LabelStatus = LabelStatus.COMPLETED,
        errors: Optional[List[str]] = None
    ) -> LabeledDataItem:
        """
        创建标注数据项。
        
        Args:
            item: 原始数据项
            labels: 标签列表
            status: 状态
            errors: 错误列表
            
        Returns:
            LabeledDataItem: 标注后的数据项
        """
        return LabeledDataItem(
            item_id=f"labeled_{getattr(item, 'data_id', id(item))}",
            original_id=getattr(item, 'data_id', ''),
            data_type=self._detect_data_type(item),
            content=self._extract_content(item),
            labels=labels,
            status=status,
            errors=errors or [],
            metadata={
                'labeler': self.__class__.__name__,
                'model': self.config.model_name,
            }
        )
    
    def _detect_data_type(self, item: Any) -> str:
        """检测数据类型"""
        if isinstance(item, str):
            return 'text'
        elif isinstance(item, dict):
            if 'image' in item or 'image_path' in item:
                return 'image'
            elif 'audio' in item or 'audio_path' in item:
                return 'audio'
            elif 'video' in item or 'video_path' in item:
                return 'video'
            elif 'text' in item:
                return 'text'
            else:
                return 'multimodal'
        else:
            return 'unknown'
    
    def _extract_content(self, item: Any) -> Any:
        """提取内容"""
        if isinstance(item, dict):
            return item.get('content') or item.get('text') or item.get('image') or item.get('audio') or item.get('video')
        else:
            return item
    
    def load_model(self):
        """加载模型（子类实现）"""
        pass
    
    def unload_model(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model unloaded")
    
    def __del__(self):
        """析构函数"""
        self.unload_model()