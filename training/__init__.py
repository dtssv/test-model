"""
训练管线模块
包含预训练、SFT、分布式训练等功能
"""

from .base_trainer import BaseTrainer, TrainingConfig, TrainingState
from .pretrainer import Pretrainer, PretrainConfig
from .sft_trainer import SFTTrainer, SFTConfig
from .distributed_trainer import DistributedTrainer, DistributedConfig

__all__ = [
    'BaseTrainer',
    'TrainingConfig',
    'TrainingState',
    'Pretrainer',
    'PretrainConfig',
    'SFTTrainer',
    'SFTConfig',
    'DistributedTrainer',
    'DistributedConfig',
]