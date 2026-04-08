"""
数据清洗管道
编排多个清洗器组成完整清洗流程
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_cleaner import BaseCleaner, CleaningConfig, CleanedDataItem, CleaningStatus

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """清洗管道阶段"""
    name: str
    cleaner: BaseCleaner
    enabled: bool = True
    order: int = 0
    parallel: bool = False  # 是否可以并行执行


@dataclass
class PipelineConfig:
    """清洗管道配置"""
    enable_parallel: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    checkpoint_interval: int = 10000
    output_dir: Optional[str] = None
    save_intermediate: bool = False


@dataclass
class PipelineStats:
    """管道统计信息"""
    total_input: int = 0
    total_output: int = 0
    total_filtered: int = 0
    total_failed: int = 0
    stage_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def update_stage(self, stage_name: str, stats: Dict[str, int]):
        """更新阶段统计"""
        self.stage_stats[stage_name] = stats


class CleaningPipeline:
    """
    数据清洗管道。
    支持串行和并行执行，支持检查点保存。
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: List[PipelineStage] = []
        self.stats = PipelineStats()
        self.logger = logging.getLogger(f"{__name__}.CleaningPipeline")
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # 检查点
        self.checkpoint_data: Dict[str, Any] = {}
    
    def add_stage(self, stage: PipelineStage) -> 'CleaningPipeline':
        """
        添加清洗阶段。
        
        Args:
            stage: 清洗阶段
            
        Returns:
            self: 支持链式调用
        """
        self.stages.append(stage)
        self.stages.sort(key=lambda s: s.order)
        return self
    
    def add_cleaner(
        self,
        name: str,
        cleaner: BaseCleaner,
        order: int = 0,
        enabled: bool = True,
        parallel: bool = False
    ) -> 'CleaningPipeline':
        """
        便捷方法：添加清洗器。
        
        Args:
            name: 阶段名称
            cleaner: 清洗器实例
            order: 执行顺序
            enabled: 是否启用
            parallel: 是否可并行
            
        Returns:
            self: 支持链式调用
        """
        stage = PipelineStage(
            name=name,
            cleaner=cleaner,
            enabled=enabled,
            order=order,
            parallel=parallel
        )
        return self.add_stage(stage)
    
    def run(self, items: List[Any]) -> List[CleanedDataItem]:
        """
        执行清洗管道。
        
        Args:
            items: 输入数据项
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项
        """
        self.logger.info(f"Starting cleaning pipeline with {len(items)} items")
        self.stats.total_input = len(items)
        
        # 当前数据
        current_items = items
        
        # 依次执行各阶段
        for stage in self.stages:
            if not stage.enabled:
                self.logger.info(f"Skipping disabled stage: {stage.name}")
                continue
            
            self.logger.info(f"Executing stage: {stage.name}")
            
            # 执行清洗
            stage_start_stats = {
                'input': len(current_items),
                'output': 0,
                'filtered': 0,
                'failed': 0
            }
            
            try:
                # 分批处理
                cleaned_items = self._process_batch(
                    stage.cleaner,
                    current_items,
                    stage.parallel
                )
                
                # 更新统计
                stage_start_stats['output'] = len([
                    i for i in cleaned_items
                    if i.status == CleaningStatus.SUCCESS
                ])
                stage_start_stats['filtered'] = len([
                    i for i in cleaned_items
                    if i.status == CleaningStatus.FILTERED
                ])
                stage_start_stats['failed'] = len([
                    i for i in cleaned_items
                    if i.status == CleaningStatus.FAILED
                ])
                
                self.stats.update_stage(stage.name, stage_start_stats)
                
                # 过滤掉被清洗的数据，保留成功和待处理的
                current_items = [
                    i for i in cleaned_items
                    if i.status in [CleaningStatus.SUCCESS, CleaningStatus.NEEDS_REVIEW]
                ]
                
                # 保存中间结果
                if self.config.save_intermediate and self.config.output_dir:
                    self._save_intermediate(stage.name, current_items)
                
                self.logger.info(
                    f"Stage {stage.name} completed: "
                    f"input={stage_start_stats['input']}, "
                    f"output={stage_start_stats['output']}, "
                    f"filtered={stage_start_stats['filtered']}, "
                    f"failed={stage_start_stats['failed']}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in stage {stage.name}: {e}")
                raise
        
        # 更新最终统计
        self.stats.total_output = len(current_items)
        self.stats.total_filtered = self.stats.total_input - self.stats.total_output
        
        self.logger.info(
            f"Pipeline completed: "
            f"input={self.stats.total_input}, "
            f"output={self.stats.total_output}, "
            f"filtered={self.stats.total_filtered}"
        )
        
        return current_items
    
    async def run_async(self, items: List[Any]) -> List[CleanedDataItem]:
        """
        异步执行清洗管道。
        
        Args:
            items: 输入数据项
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项
        """
        # 在executor中运行同步版本
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.run, items)
    
    def _process_batch(
        self,
        cleaner: BaseCleaner,
        items: List[Any],
        parallel: bool = False
    ) -> List[CleanedDataItem]:
        """
        分批处理数据。
        
        Args:
            cleaner: 清洗器
            items: 数据项
            parallel: 是否并行
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项
        """
        all_cleaned = []
        
        # 分批
        batch_size = self.config.batch_size
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        for batch_idx, batch in enumerate(batches):
            self.logger.debug(
                f"Processing batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} items)"
            )
            
            if parallel and self.config.enable_parallel:
                # 并行处理
                cleaned = self._process_parallel(cleaner, batch)
            else:
                # 串行处理
                cleaned = cleaner.clean(batch)
            
            all_cleaned.extend(cleaned)
        
        return all_cleaned
    
    def _process_parallel(
        self,
        cleaner: BaseCleaner,
        items: List[Any]
    ) -> List[CleanedDataItem]:
        """
        并行处理数据。
        
        Args:
            cleaner: 清洗器
            items: 数据项
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项
        """
        # 分割数据
        num_workers = self.config.max_workers
        chunk_size = max(1, len(items) // num_workers)
        chunks = [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        
        # 并行执行
        futures = [
            self.executor.submit(cleaner.clean, chunk)
            for chunk in chunks
        ]
        
        # 收集结果
        all_cleaned = []
        for future in futures:
            try:
                cleaned = future.result()
                all_cleaned.extend(cleaned)
            except Exception as e:
                self.logger.error(f"Error in parallel processing: {e}")
        
        return all_cleaned
    
    def _save_intermediate(self, stage_name: str, items: List[CleanedDataItem]):
        """保存中间结果"""
        import json
        from pathlib import Path
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{stage_name}_output.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([
                {
                    'item_id': item.item_id,
                    'status': item.status.value,
                    'quality_score': item.quality_score,
                    'metadata': item.metadata
                }
                for item in items
            ], f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> PipelineStats:
        """获取管道统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = PipelineStats()
    
    def enable_stage(self, stage_name: str):
        """启用阶段"""
        for stage in self.stages:
            if stage.name == stage_name:
                stage.enabled = True
                break
    
    def disable_stage(self, stage_name: str):
        """禁用阶段"""
        for stage in self.stages:
            if stage.name == stage_name:
                stage.enabled = False
                break
    
    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """获取阶段"""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None


def create_default_pipeline(
    config: Optional[PipelineConfig] = None,
    cleaner_configs: Optional[Dict[str, Any]] = None
) -> CleaningPipeline:
    """
    创建默认清洗管道。
    
    Args:
        config: 管道配置
        cleaner_configs: 清洗器配置
        
    Returns:
        CleaningPipeline: 清洗管道实例
    """
    from .text_cleaner import TextCleaner, TextCleaningConfig
    from .image_cleaner import ImageCleaner, ImageCleaningConfig
    from .dedup_engine import DedupEngine, DedupConfig
    from .pii_remover import PIIRemover, PIIConfig
    from .quality_filter import QualityFilter, QualityFilterConfig
    from .toxicity_filter import ToxicityFilter, ToxicityFilterConfig
    
    if config is None:
        config = PipelineConfig()
    
    if cleaner_configs is None:
        cleaner_configs = {}
    
    pipeline = CleaningPipeline(config)
    
    # 添加清洗阶段（按顺序）
    # 1. 文本清洗
    if 'text_cleaner' in cleaner_configs:
        text_config = TextCleaningConfig(**cleaner_configs['text_cleaner'])
    else:
        text_config = TextCleaningConfig()
    
    pipeline.add_cleaner(
        name='text_cleaning',
        cleaner=TextCleaner(text_config),
        order=1,
        parallel=True
    )
    
    # 2. 图像清洗
    if 'image_cleaner' in cleaner_configs:
        image_config = ImageCleaningConfig(**cleaner_configs['image_cleaner'])
    else:
        image_config = ImageCleaningConfig()
    
    pipeline.add_cleaner(
        name='image_cleaning',
        cleaner=ImageCleaner(image_config),
        order=2,
        parallel=True
    )
    
    # 3. 去重
    if 'dedup' in cleaner_configs:
        dedup_config = DedupConfig(**cleaner_configs['dedup'])
    else:
        dedup_config = DedupConfig()
    
    pipeline.add_cleaner(
        name='deduplication',
        cleaner=DedupEngine(dedup_config),
        order=3,
        parallel=False  # 去重需要全局状态
    )
    
    # 4. PII移除
    if 'pii' in cleaner_configs:
        pii_config = PIIConfig(**cleaner_configs['pii'])
    else:
        pii_config = PIIConfig()
    
    pipeline.add_cleaner(
        name='pii_removal',
        cleaner=PIIRemover(pii_config),
        order=4,
        parallel=True
    )
    
    # 5. 质量过滤
    if 'quality' in cleaner_configs:
        quality_config = QualityFilterConfig(**cleaner_configs['quality'])
    else:
        quality_config = QualityFilterConfig()
    
    pipeline.add_cleaner(
        name='quality_filtering',
        cleaner=QualityFilter(quality_config),
        order=5,
        parallel=True
    )
    
    # 6. 有毒内容过滤
    if 'toxicity' in cleaner_configs:
        toxicity_config = ToxicityFilterConfig(**cleaner_configs['toxicity'])
    else:
        toxicity_config = ToxicityFilterConfig()
    
    pipeline.add_cleaner(
        name='toxicity_filtering',
        cleaner=ToxicityFilter(toxicity_config),
        order=6,
        parallel=True
    )
    
    return pipeline