"""
分布式训练器
支持DeepSpeed ZeRO-3和混合精度训练
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig(TrainingConfig):
    """分布式训练配置"""
    # DeepSpeed配置
    deepspeed_config: str = None
    zero_stage: int = 3  # ZeRO阶段(0/1/2/3)
    offload_optimizer: bool = False  # 是否offload优化器到CPU
    offload_param: bool = False  # 是否offload参数到CPU
    
    # 并行配置
    local_rank: int = -1
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # 混合精度
    fp16: bool = True
    bf16: bool = False
    
    # 梯度裁剪
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        if self.deepspeed_config is None:
            self.deepspeed_config = self._generate_deepspeed_config()
    
    def _generate_deepspeed_config(self) -> str:
        """生成DeepSpeed配置"""
        config = {
            "train_batch_size": self.per_device_train_batch_size * self.world_size * self.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": self.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": self.num_train_epochs * 1000,  # 需要动态计算
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": int(0.05 * self.num_train_epochs * 1000)
                }
            },
            "gradient_clipping": self.max_grad_norm,
            "steps_per_print": self.logging_steps,
        }
        
        # 混合精度配置
        if self.fp16:
            config["fp16"] = {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        elif self.bf16:
            config["bf16"] = {
                "enabled": True
            }
        
        # ZeRO配置
        if self.zero_stage > 0:
            config["zero_optimization"] = {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none",
                    "pin_memory": True
                } if self.offload_optimizer else None,
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none",
                    "pin_memory": True
                } if self.offload_param else None,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            }
        
        # 保存配置
        config_path = Path(self.output_dir) / "deepspeed_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(config_path)


class DistributedTrainer(BaseTrainer):
    """
    分布式训练器
    支持DeepSpeed ZeRO-3、混合精度、梯度累积
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        train_dataset,
        eval_dataset=None,
    ):
        self.config = config
        self.deepspeed_engine = None
        
        super().__init__(model, config, train_dataset, eval_dataset)
        
        # 初始化分布式环境
        self._setup_distributed()
    
    def _setup_distributed(self):
        """初始化分布式训练环境"""
        try:
            import deepspeed
        except ImportError:
            raise ImportError("请安装DeepSpeed: pip install deepspeed")
        
        # 检查是否已初始化
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        
        self.local_rank = int(torch.distributed.get_rank())
        self.world_size = int(torch.distributed.get_world_size())
        
        # 设置设备
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        self.logger.info(f"Initialized distributed training: rank={self.local_rank}, world_size={self.world_size}")
    
    def _setup(self):
        """初始化（重写父类方法）"""
        # 不在父类中初始化优化器，由DeepSpeed管理
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(self.config.seed)
        
        self.logger.info("Distributed trainer initialized")
    
    def train(self):
        """执行分布式训练"""
        import deepspeed
        
        self.logger.info("Starting distributed training...")
        
        # 准备数据加载器
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        # 加载DeepSpeed配置
        with open(self.config.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        
        # 初始化DeepSpeed引擎
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config,
        )
        
        self.deepspeed_engine = model_engine
        self.optimizer = optimizer
        
        # 训练循环
        for epoch in range(self.config.num_train_epochs):
            self.state.epoch = epoch
            self._train_epoch(train_dataloader)
            
            # 评估（仅在主进程）
            if self.eval_dataset is not None and self.local_rank == 0:
                eval_metrics = self.evaluate()
                self.logger.info(f"Epoch {epoch} evaluation: {eval_metrics}")
            
            # 保存checkpoint
            if self.local_rank == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info("Distributed training completed")
        return self.state
    
    def _train_epoch(self, dataloader: DataLoader):
        """训练一个epoch"""
        self.deepspeed_engine.train()
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.deepspeed_engine(**batch)
            loss = outputs.loss
            
            # 反向传播
            self.deepspeed_engine.backward(loss)
            
            # 更新参数
            self.deepspeed_engine.step()
            
            total_loss += loss.item()
            self.state.global_step += 1
            
            # 日志记录
            if self.state.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                if self.local_rank == 0:
                    self.logger.info(f"Epoch {self.state.epoch} Step {self.state.global_step}: loss={avg_loss:.4f}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步训练（由DeepSpeed管理，不需要实现）"""
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.deepspeed_engine.eval()
        total_loss = 0.0
        num_batches = 0
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.deepspeed_engine(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.deepspeed_engine.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'eval_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int):
        """保存checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # DeepSpeed保存
        self.deepspeed_engine.save_checkpoint(
            checkpoint_dir,
            tag=f"epoch-{epoch}",
            client_state={'epoch': epoch, 'global_step': self.state.global_step}
        )
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        checkpoint_dir = Path(checkpoint_path)
        
        # DeepSpeed加载
        self.deepspeed_engine.load_checkpoint(
            checkpoint_dir,
            tag="epoch-*"
        )
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_dir}")