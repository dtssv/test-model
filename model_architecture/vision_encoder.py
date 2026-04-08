"""
视觉编码器
支持CLIP、EVA-CLIP、SigLIP等视觉模型
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class VisionEncoderConfig:
    """视觉编码器配置"""
    model_name: str = "openai/clip-vit-large-patch14-336"
    model_type: str = "clip"
    image_size: int = 336
    hidden_size: int = 1024
    freeze: bool = False


class VisionEncoder(nn.Module):
    """视觉编码器"""
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.processor = None
        self._init_model()
        
        if config.freeze:
            self._freeze_parameters()
    
    def _init_model(self):
        """初始化视觉模型"""
        from transformers import AutoModel, AutoImageProcessor
        
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
    
    def _freeze_parameters(self):
        """冻结参数"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': getattr(outputs, 'pooler_output', None),
        }
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        outputs = self.forward(images)
        return outputs['last_hidden_state']
    
    def get_hidden_size(self) -> int:
        return self.config.hidden_size


def create_vision_encoder(model_name: str, freeze: bool = False) -> VisionEncoder:
    """创建视觉编码器"""
    config = VisionEncoderConfig(model_name=model_name, freeze=freeze)
    return VisionEncoder(config)