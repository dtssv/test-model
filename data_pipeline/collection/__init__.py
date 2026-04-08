"""
数据采集模块
"""

from .base_collector import BaseCollector, CollectionConfig, DataSource, RawDataItem, BatchSaveResult
from .text_collector import TextCollector, TextCollectionConfig
from .image_text_collector import ImageTextCollector, ImageTextCollectionConfig
from .video_collector import VideoCollector, VideoCollectionConfig
from .audio_collector import AudioCollector, AudioCollectionConfig
from .deduplicator import Deduplicator, DedupConfig

__all__ = [
    'BaseCollector', 'CollectionConfig', 'DataSource', 'RawDataItem', 'BatchSaveResult',
    'TextCollector', 'TextCollectionConfig',
    'ImageTextCollector', 'ImageTextCollectionConfig',
    'VideoCollector', 'VideoCollectionConfig',
    'AudioCollector', 'AudioCollectionConfig',
    'Deduplicator', 'DedupConfig',
]