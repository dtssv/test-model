"""
模态投影器
将视觉、音频编码映射到语言模型空间
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ProjectorConfig:
    """投影器配置"""
    # 输入输出维度
    input_dim: int = 1024
    output_dim: int = 4096
    
    # 投影器类型
    projector_type: str = "mlp"  # mlp, qformer, resampler, linear
    
    # MLP配置
    mlp_hidden_dim: int = 4096
    mlp_depth: int = 2
    mlp_activation: str = "gelu"
    mlp_dropout: float = 0.1
    
    # Q-Former配置
    qformer_num_queries: int = 32
    qformer_num_heads: int = 8
    qformer_num_layers: int = 2
    
    # Resampler配置
    resampler_num_queries: int = 64
    resampler_depth: int = 3
    
    # 其他
    use_layer_norm: bool = True
    bias: bool = True


class MLPProjector(nn.Module):
    """MLP投影器"""
    
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.input_dim
        
        for i in range(config.mlp_depth):
            out_dim = config.mlp_hidden_dim if i < config.mlp_depth - 1 else config.output_dim
            
            # 线性层
            layers.append(nn.Linear(in_dim, out_dim, bias=config.bias))
            
            # Layer Norm
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            # 激活函数
            if i < config.mlp_depth - 1:
                if config.mlp_activation == "gelu":
                    layers.append(nn.GELU())
                elif config.mlp_activation == "relu":
                    layers.append(nn.ReLU())
                elif config.mlp_activation == "silu":
                    layers.append(nn.SiLU())
                
                # Dropout
                if config.mlp_dropout > 0:
                    layers.append(nn.Dropout(config.mlp_dropout))
            
            in_dim = out_dim
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.projector(x)


class QFormerProjector(nn.Module):
    """Q-Former投影器"""
    
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        
        # 查询向量
        self.queries = nn.Parameter(
            torch.randn(config.qformer_num_queries, config.output_dim)
        )
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=config.qformer_num_heads,
            batch_first=True,
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim * 4),
            nn.GELU(),
            nn.Linear(config.output_dim * 4, config.output_dim),
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(config.output_dim)
        self.norm2 = nn.LayerNorm(config.output_dim)
        
        # 输入投影
        self.input_proj = nn.Linear(config.input_dim, config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.shape[0]
        
        # 扩展查询
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 投影输入
        x_proj = self.input_proj(x)
        
        # 交叉注意力
        attn_out, _ = self.cross_attention(
            query=queries,
            key=x_proj,
            value=x_proj,
        )
        queries = self.norm1(queries + attn_out)
        
        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        
        return queries


class ResamplerProjector(nn.Module):
    """Resampler投影器"""
    
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        
        # 查询向量
        self.queries = nn.Parameter(
            torch.randn(config.resampler_num_queries, config.output_dim)
        )
        
        # Latent Transformer层
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.output_dim,
                nhead=8,
                dim_feedforward=config.output_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(config.resampler_depth)
        ])
        
        # 输入投影
        self.input_proj = nn.Linear(config.input_dim, config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.shape[0]
        
        # 扩展查询
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 投影输入
        memory = self.input_proj(x)
        
        # Transformer层
        for layer in self.layers:
            queries = layer(tgt=queries, memory=memory)
        
        return queries


class LinearProjector(nn.Module):
    """线性投影器"""
    
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        self.projector = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)
        
        if config.use_layer_norm:
            self.norm = nn.LayerNorm(config.output_dim)
        else:
            self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.projector(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class MultimodalProjector(nn.Module):
    """多模态投影器"""
    
    PROJECTOR_CLASSES = {
        "mlp": MLPProjector,
        "qformer": QFormerProjector,
        "resampler": ResamplerProjector,
        "linear": LinearProjector,
    }
    
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        
        projector_class = self.PROJECTOR_CLASSES.get(config.projector_type)
        if projector_class is None:
            raise ValueError(f"Unknown projector type: {config.projector_type}")
        
        self.projector = projector_class(config)
        
        logger.info(f"Initialized {config.projector_type} projector")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            
        Returns:
            投影后特征 [batch_size, new_seq_len, output_dim]
        """
        return self.projector(x)
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.config.output_dim


def create_projector(
    input_dim: int,
    output_dim: int,
    projector_type: str = "mlp",
    **kwargs
) -> MultimodalProjector:
    """
    创建投影器的便捷函数
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        projector_type: 投影器类型
        
    Returns:
        MultimodalProjector: 投影器实例
    """
    config = ProjectorConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        projector_type=projector_type,
        **kwargs
    )
    return MultimodalProjector(config)