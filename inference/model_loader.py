"""
模型加载器
支持加载多模态模型及其各组件
"""

from typing import Optional, Dict, Any
import logging
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    模型加载器
    支持从预训练checkpoint加载多模态模型
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型路径
            device: 设备类型 (cuda/cpu)
            dtype: 数据类型 (float16/bfloat16/float32)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = self._get_dtype(dtype)
        
        self.logger = logging.getLogger(__name__)
    
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """获取数据类型"""
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def load_model(
        self,
        model_class: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        load_weights: bool = True,
    ) -> nn.Module:
        """
        加载模型
        
        Args:
            model_class: 模型类
            config: 模型配置
            load_weights: 是否加载权重
        
        Returns:
            加载的模型
        """
        self.logger.info(f"Loading model from {self.model_path}")
        
        # 初始化模型
        if config is None:
            config = self.load_config()
        
        model = model_class(**config)
        
        # 加载权重
        if load_weights:
            self.load_weights(model)
        
        # 转换数据类型并移动到设备
        model = model.to(self.dtype)
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        
        self.logger.info(f"Model loaded successfully")
        return model
    
    def load_weights(self, model: nn.Module):
        """
        加载模型权重
        
        Args:
            model: 模型实例
        """
        # 检查权重文件
        weights_files = list(self.model_path.glob("*.bin")) + \
                       list(self.model_path.glob("*.pt")) + \
                       list(self.model_path.glob("*.safetensors"))
        
        if not weights_files:
            self.logger.warning(f"No weight files found in {self.model_path}")
            return
        
        # 加载权重
        for weights_file in weights_files:
            if weights_file.suffix == ".safetensors":
                self._load_safetensors(model, weights_file)
            else:
                self._load_pytorch(model, weights_file)
        
        self.logger.info(f"Model weights loaded from {self.model_path}")
    
    def _load_pytorch(self, model: nn.Module, weights_file: Path):
        """加载PyTorch权重"""
        state_dict = torch.load(weights_file, map_location="cpu")
        
        # 处理可能的键名不匹配
        model_state_dict = model.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # 移除可能的前缀
            name = k
            if k.startswith("model."):
                name = k[6:]
            elif k.startswith("module."):
                name = k[7:]
            
            if name in model_state_dict:
                new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    
    def _load_safetensors(self, model: nn.Module, weights_file: Path):
        """加载SafeTensors权重"""
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weights_file)
            model.load_state_dict(state_dict, strict=False)
        except ImportError:
            self.logger.warning("safetensors库未安装，跳过加载")
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载模型配置
        
        Returns:
            配置字典
        """
        config_file = self.model_path / "config.json"
        
        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_file}")
            return {}
        
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    def load_tokenizer(self):
        """
        加载tokenizer
        
        Returns:
            tokenizer实例
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.logger.info(f"Tokenizer loaded from {self.model_path}")
            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            return None
    
    def load_vision_encoder(self) -> Optional[nn.Module]:
        """
        加载视觉编码器
        
        Returns:
            视觉编码器实例
        """
        try:
            from transformers import AutoModel
            vision_path = self.model_path / "vision_encoder"
            
            if not vision_path.exists():
                self.logger.warning(f"Vision encoder not found at {vision_path}")
                return None
            
            vision_encoder = AutoModel.from_pretrained(vision_path)
            vision_encoder = vision_encoder.to(self.dtype)
            
            if self.device == "cuda":
                vision_encoder = vision_encoder.cuda()
            
            self.logger.info(f"Vision encoder loaded from {vision_path}")
            return vision_encoder
        except Exception as e:
            self.logger.error(f"Failed to load vision encoder: {e}")
            return None
    
    @staticmethod
    def from_pretrained(
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> "ModelLoader":
        """
        从预训练模型创建加载器
        
        Args:
            model_path: 模型路径
            device: 设备类型
            dtype: 数据类型
        
        Returns:
            ModelLoader实例
        """
        return ModelLoader(model_path, device, dtype)