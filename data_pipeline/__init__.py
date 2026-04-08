"""
数据管线模块
提供多模态大模型数据的完整处理流程
"""

# 采集模块
from .collection.base_collector import (
    BaseCollector,
    CollectionConfig,
    DataSource,
    RawDataItem,
    BatchSaveResult,
)

from .collection.text_collector import (
    TextCollector,
    TextCollectorConfig,
)

from .collection.image_text_collector import (
    ImageTextCollector,
    ImageTextCollectorConfig,
)

# 存储模块
from .storage.minio_client import (
    MinIOClient,
    ObjectInfo,
)

from .storage.metadata_store import (
    MetadataStore,
    DatasetInfo,
    DataItemMeta,
    DatasetStats,
)

# 清洗模块
from .cleaning import (
    BaseCleaner,
    CleaningConfig,
    CleanedDataItem,
    CleaningStats,
    CleaningStatus,
    TextCleaner,
    TextCleaningConfig,
    ImageCleaner,
    ImageCleaningConfig,
    DedupEngine,
    DedupConfig,
    PIIRemover,
    PIIConfig,
    QualityFilter,
    QualityFilterConfig,
    ToxicityFilter,
    ToxicityFilterConfig,
    CleaningPipeline,
    PipelineConfig,
    create_default_pipeline,
)

# 打标模块
from .labeling import (
    BaseLabeler,
    LabelConfig,
    Label,
    LabelType,
    LabeledDataItem,
    LabelStatus,
    LabelingStats,
    CaptionLabeler,
    CaptionLabelerConfig,
    QALabeler,
    QALabelerConfig,
    QualityScorer,
    QualityScorerConfig,
    SafetyLabeler,
    SafetyLabelerConfig,
)

# Tokenization模块
from .tokenization import (
    BaseTokenizer,
    TokenizerConfig,
    TokenizedOutput,
    TokenizationStats,
    ModalityType,
    TextTokenizer,
    TextTokenizerConfig,
    MultimodalTokenizer,
    MultimodalTokenizerConfig,
)

# 管线编排
from .pipeline import (
    DataPipeline,
    DataPipelineConfig,
    PipelineStats,
    create_pipeline,
)

__all__ = [
    # 采集
    'BaseCollector',
    'CollectionConfig',
    'DataSource',
    'RawDataItem',
    'BatchSaveResult',
    'TextCollector',
    'TextCollectorConfig',
    'ImageTextCollector',
    'ImageTextCollectorConfig',
    
    # 存储
    'MinIOClient',
    'ObjectInfo',
    'MetadataStore',
    'DatasetInfo',
    'DataItemMeta',
    'DatasetStats',
    
    # 清洗
    'BaseCleaner',
    'CleaningConfig',
    'CleanedDataItem',
    'CleaningStats',
    'CleaningStatus',
    'TextCleaner',
    'TextCleaningConfig',
    'ImageCleaner',
    'ImageCleaningConfig',
    'DedupEngine',
    'DedupConfig',
    'PIIRemover',
    'PIIConfig',
    'QualityFilter',
    'QualityFilterConfig',
    'ToxicityFilter',
    'ToxicityFilterConfig',
    'CleaningPipeline',
    'PipelineConfig',
    'create_default_pipeline',
    
    # 打标
    'BaseLabeler',
    'LabelConfig',
    'Label',
    'LabelType',
    'LabeledDataItem',
    'LabelStatus',
    'LabelingStats',
    'CaptionLabeler',
    'CaptionLabelerConfig',
    'QALabeler',
    'QALabelerConfig',
    'QualityScorer',
    'QualityScorerConfig',
    'SafetyLabeler',
    'SafetyLabelerConfig',
    
    # Tokenization
    'BaseTokenizer',
    'TokenizerConfig',
    'TokenizedOutput',
    'TokenizationStats',
    'ModalityType',
    'TextTokenizer',
    'TextTokenizerConfig',
    'MultimodalTokenizer',
    'MultimodalTokenizerConfig',
    
    # 管线
    'DataPipeline',
    'DataPipelineConfig',
    'PipelineStats',
    'create_pipeline',
]