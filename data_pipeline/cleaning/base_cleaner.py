"""
数据清洗器基类
定义所有清洗器的标准接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CleaningStatus(Enum):
    """清洗状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    FILTERED = "filtered"  # 被过滤掉
    SKIPPED = "skipped"


@dataclass
class CleaningConfig:
    """数据清洗通用配置"""
    batch_size: int = 100
    num_workers: int = 4
    enable_parallel: bool = True
    quality_threshold: float = 0.5
    remove_duplicates: bool = True
    remove_pii: bool = True
    filter_low_quality: bool = True
    output_format: str = "jsonl"  # jsonl, parquet, binary
    max_retries: int = 3


@dataclass
class CleanedDataItem:
    """清洗后的数据项"""
    item_id: str
    original_id: str
    data_type: str
    content: Any
    status: CleaningStatus
    quality_score: Optional[float] = None
    cleaning_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    cleaned_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'original_id': self.original_id,
            'data_type': self.data_type,
            'status': self.status.value,
            'quality_score': self.quality_score,
            'cleaning_steps': self.cleaning_steps,
            'metadata': self.metadata,
            'errors': self.errors,
            'cleaned_at': self.cleaned_at.isoformat()
        }


@dataclass
class CleaningResult:
    """批量清洗结果"""
    total_items: int
    success_items: int
    failed_items: int
    filtered_items: int
    items: List[CleanedDataItem] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.success_items / self.total_items


class BaseCleaner(ABC):
    """
    数据清洗器抽象基类。
    定义清洗、验证、过滤的标准接口。
    """
    
    def __init__(self, config: CleaningConfig):
        """
        初始化清洗器
        
        Args:
            config: 清洗配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cleaning_stats = {
            'total_processed': 0,
            'success': 0,
            'failed': 0,
            'filtered': 0
        }
    
    @abstractmethod
    def clean(self, items: List[Any]) -> List[CleanedDataItem]:
        """
        执行清洗流程
        
        Args:
            items: 待清洗的数据项列表
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项列表
        """
        pass
    
    @abstractmethod
    def validate(self, item: Any) -> bool:
        """
        验证数据项是否有效
        
        Args:
            item: 待验证的数据项
            
        Returns:
            bool: 是否通过验证
        """
        pass
    
    def filter_quality(self, item: Any, threshold: float = None) -> bool:
        """
        过滤低质量数据
        
        Args:
            item: 数据项
            threshold: 质量阈值
            
        Returns:
            bool: 是否保留
        """
        if threshold is None:
            threshold = self.config.quality_threshold
        
        # 默认实现，子类可以覆盖
        return True
    
    def batch_clean(self, items: List[Any], batch_size: int = None) -> CleaningResult:
        """
        批量清洗数据
        
        Args:
            items: 数据项列表
            batch_size: 批次大小
            
        Returns:
            CleaningResult: 清洗结果
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        cleaned_items = []
        success_count = 0
        failed_count = 0
        filtered_count = 0
        
        # 分批处理
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                # 清洗批次
                batch_result = self.clean(batch)
                
                for item in batch_result:
                    cleaned_items.append(item)
                    
                    if item.status == CleaningStatus.SUCCESS:
                        success_count += 1
                    elif item.status == CleaningStatus.FAILED:
                        failed_count += 1
                    elif item.status == CleaningStatus.FILTERED:
                        filtered_count += 1
                
                # 更新统计
                self.cleaning_stats['total_processed'] += len(batch)
                self.cleaning_stats['success'] += sum(1 for item in batch_result if item.status == CleaningStatus.SUCCESS)
                self.cleaning_stats['failed'] += sum(1 for item in batch_result if item.status == CleaningStatus.FAILED)
                self.cleaning_stats['filtered'] += sum(1 for item in batch_result if item.status == CleaningStatus.FILTERED)
                
                self.logger.info(
                    f"Processed batch {i//batch_size + 1}: "
                    f"{len(batch_result)} items, "
                    f"success={success_count}, failed={failed_count}, filtered={filtered_count}"
                )
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # 将失败的项目标记为失败
                for item in batch:
                    if hasattr(item, 'data_id'):
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{item.data_id}",
                            original_id=item.data_id,
                            data_type='unknown',
                            content=None,
                            status=CleaningStatus.FAILED,
                            errors=[str(e)]
                        ))
                        failed_count += 1
        
        # 生成统计信息
        statistics = {
            'total_items': len(items),
            'success_rate': success_count / len(items) if items else 0,
            'filter_rate': filtered_count / len(items) if items else 0,
            'average_quality_score': sum(
                item.quality_score for item in cleaned_items 
                if item.quality_score is not None
            ) / success_count if success_count > 0 else 0
        }
        
        return CleaningResult(
            total_items=len(items),
            success_items=success_count,
            failed_items=failed_count,
            filtered_items=filtered_count,
            items=cleaned_items,
            statistics=statistics
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取清洗统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.cleaning_stats.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.cleaning_stats = {
            'total_processed': 0,
            'success': 0,
            'failed': 0,
            'filtered': 0
        }
    
    def log_summary(self):
        """输出清洗摘要"""
        stats = self.cleaning_stats
        total = stats['total_processed']
        
        if total > 0:
            self.logger.info(
                f"Cleaning Summary - "
                f"Total: {total}, "
                f"Success: {stats['success']} ({stats['success']/total*100:.2f}%), "
                f"Failed: {stats['failed']} ({stats['failed']/total*100:.2f}%), "
                f"Filtered: {stats['filtered']} ({stats['filtered']/total*100:.2f}%)"
            )
        else:
            self.logger.info("No items processed")