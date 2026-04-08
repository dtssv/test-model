"""
模型架构模块
提供多模态大模型的编码器、投影器、骨干网络等组件
"""

from .vision_encoder import (
    VisionEncoder,
    VisionEncoderConfig,
    create_vision_encoder,
)

from .projector import (
    MultimodalProjector,
    ProjectorConfig,
    MLPProjector,
    QFormerProjector,
    ResamplerProjector,
    LinearProjector,
    create_projector,
)

from .llm_backbone import (
    LLMBackbone,
    LLMConfig,
    RMSNorm,
    Attention,
    MLP,
    DecoderLayer,
    create_llm_backbone,
)

__all__ = [
    # 视觉编码器
    'VisionEncoder',
    'VisionEncoderConfig',
    'create_vision_encoder',
    
    # 投影器
    'MultimodalProjector',
    'ProjectorConfig',
    'MLPProjector',
    'QFormerProjector',
    'ResamplerProjector',
    'LinearProjector',
    'create_projector',
    
    # 语言模型骨干
    'LLMBackbone',
    'LLMConfig',
    'RMSNorm',
    'Attention',
    'MLP',
    'DecoderLayer',
    'create_llm_backbone',
]