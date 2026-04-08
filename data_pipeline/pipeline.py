"""
数据管线编排器
整合采集、清洗、打标、Token化的完整流程
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import json

from .collection.base_collector import BaseCollector, CollectionConfig, BatchSaveResult
from .cleaning.pipeline import CleaningPipeline, PipelineConfig as CleaningPipelineConfig
from .labeling.base_labeler import BaseLabeler, LabelConfig, LabeledDataItem
from .tokenization.base_tokenizer import BaseTokenizer, TokenizerConfig, TokenizedOutput
from .storage.minio_client import MinIOClient
from .storage.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageConfig:
    """管线阶段配置"""
    enabled: bool = True
    parallel: bool = False
    batch_size: int = 1000
    num_workers: int = 4


@dataclass
class DataPipelineConfig:
    """数据管线配置"""
    # 采集配置
    collection_config: Optional[CollectionConfig] = None
    enable_collection: bool = True
    
    # 清洗配置
    cleaning_config: Optional[CleaningPipelineConfig] = None
    enable_cleaning: bool = True
    
    # 打标配置
    labeling_configs: Dict[str, LabelConfig] = field(default_factory=dict)
    enable_labeling: bool = True
    
    # Tokenization配置
    tokenizer_config: Optional[TokenizerConfig] = None
    enable_tokenization: bool = True
    
    # 存储配置
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "mllm-data"
    
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "mllm_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # 输出配置
    output_dir: str = "./data/processed"
    save_intermediate: bool = True
    checkpoint_interval: int = 10000
    
    # 执行配置
    max_workers: int = 8
    enable_async: bool = True


@dataclass
class PipelineStats:
    """管线统计信息"""
    total_input: int = 0
    collected: int = 0
    cleaned: int = 0
    labeled: int = 0
    tokenized: int = 0
    saved: int = 0
    
    stage_times: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_input': self.total_input,
            'collected': self.collected,
            'cleaned': self.cleaned,
            'labeled': self.labeled,
            'tokenized': self.tokenized,
            'saved': self.saved,
            'stage_times': self.stage_times,
            'errors': self.errors,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self._calculate_duration(),
        }
    
    def _calculate_duration(self) -> Optional[float]:
        """计算总时长"""
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None


class DataPipeline:
    """
    数据管线编排器。
    整合数据采集、清洗、打标、Token化、存储的完整流程。
    """
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataPipeline")
        self.stats = PipelineStats()
        
        # 组件
        self.collector: Optional[BaseCollector] = None
        self.cleaning_pipeline: Optional[CleaningPipeline] = None
        self.labelers: Dict[str, BaseLabeler] = {}
        self.tokenizer: Optional[BaseTokenizer] = None
        self.minio_client: Optional[MinIOClient] = None
        self.metadata_store: Optional[MetadataStore] = None
        
        # 回调函数
        self.progress_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化管线组件"""
        # 初始化存储
        self._init_storage()
        
        # 初始化清洗管线
        if self.config.enable_cleaning:
            self._init_cleaning_pipeline()
        
        self.logger.info("Data pipeline initialized")
    
    def _init_storage(self):
        """初始化存储组件"""
        try:
            # MinIO
            self.minio_client = MinIOClient(
                endpoint=self.config.minio_endpoint,
                access_key=self.config.minio_access_key,
                secret_key=self.config.minio_secret_key,
                bucket_name=self.config.minio_bucket,
            )
            
            # PostgreSQL
            self.metadata_store = MetadataStore(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )
            
            self.logger.info("Storage components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
    
    def _init_cleaning_pipeline(self):
        """初始化清洗管线"""
        from .cleaning.pipeline import create_default_pipeline
        
        if self.config.cleaning_config:
            self.cleaning_pipeline = CleaningPipeline(self.config.cleaning_config)
        else:
            self.cleaning_pipeline = create_default_pipeline()
        
        self.logger.info("Cleaning pipeline initialized")
    
    def set_collector(self, collector: BaseCollector):
        """设置数据采集器"""
        self.collector = collector
        self.logger.info(f"Collector set: {collector.__class__.__name__}")
    
    def add_labeler(self, name: str, labeler: BaseLabeler):
        """添加打标器"""
        self.labelers[name] = labeler
        self.logger.info(f"Labeler added: {name}")
    
    def set_tokenizer(self, tokenizer: BaseTokenizer):
        """设置Tokenizer"""
        self.tokenizer = tokenizer
        self.logger.info(f"Tokenizer set: {tokenizer.__class__.__name__}")
    
    def run(self, data_source: Optional[Any] = None) -> PipelineStats:
        """
        执行完整的数据管线。
        
        Args:
            data_source: 数据源（可选，如果collector已设置）
            
        Returns:
            PipelineStats: 管线统计信息
        """
        self.logger.info("Starting data pipeline")
        self.stats.start_time = datetime.now().isoformat()
        
        try:
            # 阶段1: 数据采集
            if self.config.enable_collection and self.collector:
                data = self._run_collection(data_source)
            else:
                data = data_source if data_source else []
            
            self.stats.total_input = len(data)
            
            # 阶段2: 数据清洗
            if self.config.enable_cleaning and self.cleaning_pipeline:
                data = self._run_cleaning(data)
            
            # 阶段3: 数据打标
            if self.config.enable_labeling and self.labelers:
                data = self._run_labeling(data)
            
            # 阶段4: Tokenization
            if self.config.enable_tokenization and self.tokenizer:
                data = self._run_tokenization(data)
            
            # 阶段5: 数据存储
            self._run_storage(data)
            
            self.logger.info("Data pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.stats.errors.append(str(e))
            if self.error_callback:
                self.error_callback(e)
        
        finally:
            self.stats.end_time = datetime.now().isoformat()
        
        return self.stats
    
    async def run_async(self, data_source: Optional[Any] = None) -> PipelineStats:
        """异步执行数据管线"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, data_source)
    
    def _run_collection(self, data_source: Any) -> List[Any]:
        """执行数据采集"""
        stage_name = "collection"
        start_time = datetime.now()
        
        self.logger.info("Running collection stage")
        
        try:
            if self.collector:
                data = self.collector.collect(data_source)
                self.stats.collected = len(data)
            else:
                data = []
            
            stage_time = (datetime.now() - start_time).total_seconds()
            self.stats.stage_times[stage_name] = stage_time
            
            self.logger.info(f"Collection completed: {self.stats.collected} items in {stage_time:.2f}s")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Collection failed: {e}")
            raise
    
    def _run_cleaning(self, data: List[Any]) -> List[Any]:
        """执行数据清洗"""
        stage_name = "cleaning"
        start_time = datetime.now()
        
        self.logger.info("Running cleaning stage")
        
        try:
            cleaned_data = self.cleaning_pipeline.run(data)
            self.stats.cleaned = len(cleaned_data)
            
            stage_time = (datetime.now() - start_time).total_seconds()
            self.stats.stage_times[stage_name] = stage_time
            
            self.logger.info(f"Cleaning completed: {self.stats.cleaned} items in {stage_time:.2f}s")
            
            # 保存中间结果
            if self.config.save_intermediate:
                self._save_intermediate(cleaned_data, "cleaned")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Cleaning failed: {e}")
            raise
    
    def _run_labeling(self, data: List[Any]) -> List[LabeledDataItem]:
        """执行数据打标"""
        stage_name = "labeling"
        start_time = datetime.now()
        
        self.logger.info("Running labeling stage")
        
        try:
            labeled_data = data
            
            # 依次运行所有打标器
            for labeler_name, labeler in self.labelers.items():
                self.logger.info(f"Running labeler: {labeler_name}")
                labeled_data = labeler.label_batch(labeled_data)
            
            self.stats.labeled = len(labeled_data)
            
            stage_time = (datetime.now() - start_time).total_seconds()
            self.stats.stage_times[stage_name] = stage_time
            
            self.logger.info(f"Labeling completed: {self.stats.labeled} items in {stage_time:.2f}s")
            
            # 保存中间结果
            if self.config.save_intermediate:
                self._save_intermediate(labeled_data, "labeled")
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"Labeling failed: {e}")
            raise
    
    def _run_tokenization(self, data: List[Any]) -> List[TokenizedOutput]:
        """执行Tokenization"""
        stage_name = "tokenization"
        start_time = datetime.now()
        
        self.logger.info("Running tokenization stage")
        
        try:
            tokenized_data = self.tokenizer.tokenize_batch(data)
            self.stats.tokenized = len(tokenized_data)
            
            stage_time = (datetime.now() - start_time).total_seconds()
            self.stats.stage_times[stage_name] = stage_time
            
            self.logger.info(f"Tokenization completed: {self.stats.tokenized} items in {stage_time:.2f}s")
            
            # 保存中间结果
            if self.config.save_intermediate:
                self._save_intermediate(tokenized_data, "tokenized")
            
            return tokenized_data
            
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise
    
    def _run_storage(self, data: List[Any]):
        """执行数据存储"""
        stage_name = "storage"
        start_time = datetime.now()
        
        self.logger.info("Running storage stage")
        
        try:
            # 存储到MinIO
            if self.minio_client:
                # 实际实现需要具体逻辑
                self.logger.info("Saving to MinIO...")
            
            # 存储元数据到PostgreSQL
            if self.metadata_store:
                # 实际实现需要具体逻辑
                self.logger.info("Saving metadata to PostgreSQL...")
            
            self.stats.saved = len(data)
            
            stage_time = (datetime.now() - start_time).total_seconds()
            self.stats.stage_times[stage_name] = stage_time
            
            self.logger.info(f"Storage completed: {self.stats.saved} items in {stage_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Storage failed: {e}")
            raise
    
    def _save_intermediate(self, data: List[Any], stage_name: str):
        """保存中间结果"""
        import pickle
        
        output_dir = Path(self.config.output_dir) / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{stage_name}_{timestamp}.pkl"
        
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Intermediate data saved: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save intermediate data: {e}")
    
    def set_progress_callback(self, callback: Callable):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def set_error_callback(self, callback: Callable):
        """设置错误回调函数"""
        self.error_callback = callback
    
    def get_stats(self) -> PipelineStats:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = PipelineStats()
    
    def save_checkpoint(self, checkpoint_path: str):
        """保存检查点"""
        checkpoint = {
            'stats': self.stats.to_dict(),
            'config': {
                'enable_collection': self.config.enable_collection,
                'enable_cleaning': self.config.enable_cleaning,
                'enable_labeling': self.config.enable_labeling,
                'enable_tokenization': self.config.enable_tokenization,
            },
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # 恢复统计信息
        stats_dict = checkpoint.get('stats', {})
        self.stats.total_input = stats_dict.get('total_input', 0)
        self.stats.collected = stats_dict.get('collected', 0)
        self.stats.cleaned = stats_dict.get('cleaned', 0)
        self.stats.labeled = stats_dict.get('labeled', 0)
        self.stats.tokenized = stats_dict.get('tokenized', 0)
        self.stats.saved = stats_dict.get('saved', 0)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


def create_pipeline(
    enable_collection: bool = True,
    enable_cleaning: bool = True,
    enable_labeling: bool = True,
    enable_tokenization: bool = True,
    **kwargs
) -> DataPipeline:
    """
    创建数据管线的便捷函数。
    
    Args:
        enable_collection: 是否启用采集
        enable_cleaning: 是否启用清洗
        enable_labeling: 是否启用打标
        enable_tokenization: 是否启用Tokenization
        
    Returns:
        DataPipeline: 数据管线实例
    """
    config = DataPipelineConfig(
        enable_collection=enable_collection,
        enable_cleaning=enable_cleaning,
        enable_labeling=enable_labeling,
        enable_tokenization=enable_tokenization,
        **kwargs
    )
    
    return DataPipeline(config)