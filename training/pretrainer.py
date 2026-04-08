"""
预训练训练器
实现多模态预训练的三阶段训练流程
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .base_trainer import BaseTrainer, TrainingConfig, TrainingState

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig(TrainingConfig):
    """预训练配置"""
    stage: int = 1  # 训练阶段(1/2/3)
    vision_encoder_lr: float = 1e-5  # 视觉编码器学习率
    llm_lr: float = 5e-5  # 语言模型学习率
    projector_lr: float = 1e-4  # 投影器学习率
    warmup_ratio: float = 0.05  # warmup比例
    max_grad_norm: float = 1.0
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    
    # 阶段特定配置
    freeze_vision_encoder: bool = True  # 冻结视觉编码器
    freeze_llm: bool = False  # 冻结语言模型
    unfreeze_llm_layers: int = -1  # 解冻LLM最后N层(-1表示全部解冻)
    unfreeze_vision_layers: int = 0  # 解冻视觉编码器最后N层


class Pretrainer(BaseTrainer):
    """
    多模态预训练器
    阶段1: 冻结视觉编码器和LLM，仅训练投影器(模态对齐)
    阶段2: 解冻LLM，联合训练投影器和LLM(知识注入)
    阶段3: 可选解冻视觉编码器最后N层，全量微调
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PretrainConfig,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
    ):
        super().__init__(model, config, train_dataset, eval_dataset)
        self.config = config
        self.tokenizer = tokenizer
        self.lr_scheduler = None
        
        self._configure_trainable_parameters()
    
    def _configure_trainable_parameters(self):
        """根据训练阶段配置可训练参数"""
        # 获取模型各组件
        vision_encoder = getattr(self.model, 'vision_encoder', None)
        llm_backbone = getattr(self.model, 'llm_backbone', None)
        projector = getattr(self.model, 'projector', None)
        
        # 阶段1: 仅训练投影器
        if self.config.stage == 1:
            self._freeze_module(vision_encoder)
            self._freeze_module(llm_backbone)
            self._unfreeze_module(projector)
            self.logger.info("Stage 1: Training projector only")
        
        # 阶段2: 训练投影器和LLM
        elif self.config.stage == 2:
            self._freeze_module(vision_encoder)
            self._unfreeze_module(llm_backbone, self.config.unfreeze_llm_layers)
            self._unfreeze_module(projector)
            self.logger.info("Stage 2: Training projector + LLM")
        
        # 阶段3: 全量训练
        elif self.config.stage == 3:
            self._unfreeze_module(vision_encoder, self.config.unfreeze_vision_layers)
            self._unfreeze_module(llm_backbone, self.config.unfreeze_llm_layers)
            self._unfreeze_module(projector)
            self.logger.info("Stage 3: Full fine-tuning")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")
    
    def _freeze_module(self, module: Optional[nn.Module]):
        """冻结模块参数"""
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
    
    def _unfreeze_module(self, module: Optional[nn.Module], last_n_layers: int = -1):
        """解冻模块参数"""
        if module is None:
            return
        
        if last_n_layers == -1:
            # 解冻全部
            for param in module.parameters():
                param.requires_grad = True
        else:
            # 解冻最后N层
            layers = self._get_layers(module)
            for i, layer in enumerate(layers):
                if i >= len(layers) - last_n_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
    
    def _get_layers(self, module: nn.Module) -> List[nn.Module]:
        """获取模块的层列表"""
        if hasattr(module, 'layers'):
            return list(module.layers)
        elif hasattr(module, 'encoder') and hasattr(module.encoder, 'layers'):
            return list(module.encoder.layers)
        else:
            return [module]
    
    def setup_optimizer(self) -> Optimizer:
        """创建优化器，不同参数组使用不同学习率"""
        # 定义参数组
        param_groups = []
        
        # 投影器参数
        projector_params = [p for n, p in self.model.named_parameters() 
                           if 'projector' in n and p.requires_grad]
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': self.config.projector_lr
            })
        
        # LLM参数
        llm_params = [p for n, p in self.model.named_parameters() 
                     if 'llm' in n and p.requires_grad]
        if llm_params:
            param_groups.append({
                'params': llm_params,
                'lr': self.config.llm_lr
            })
        
        # 视觉编码器参数
        vision_params = [p for n, p in self.model.named_parameters() 
                        if 'vision' in n and p.requires_grad]
        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': self.config.vision_encoder_lr
            })
        
        # 其他参数
        other_params = [p for n, p in self.model.named_parameters() 
                       if p.requires_grad and 
                       'projector' not in n and 'llm' not in n and 'vision' not in n]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate
            })
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )
        
        return optimizer
    
    def setup_lr_scheduler(self, num_training_steps: int) -> LRScheduler:
        """设置学习率调度器: 线性warmup + 余弦衰减"""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return scheduler
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步训练"""
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss
    
    def train(self) -> TrainingState:
        """执行训练"""
        self.logger.info(f"Starting pretraining stage {self.config.stage}...")
        
        # 设置数据加载器
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        # 设置优化器和学习率调度器
        self.optimizer = self.setup_optimizer()
        num_training_steps = len(train_dataloader) * self.config.num_train_epochs
        self.lr_scheduler = self.setup_lr_scheduler(num_training_steps)
        
        # 训练循环
        for epoch in range(self.config.num_train_epochs):
            self.state.epoch = epoch
            self._train_epoch(train_dataloader)
            
            # 评估
            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                self.logger.info(f"Epoch {epoch} evaluation: {eval_metrics}")
            
            # 保存checkpoint
            self.save_checkpoint(epoch)
        
        self.logger.info("Pretraining completed")
        return self.state
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
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
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'eval_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int):
        """保存checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_dir)
        
        # 保存优化器和调度器
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        if self.lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # 保存训练状态
        state_dict = {
            'epoch': epoch,
            'global_step': self.state.global_step,
            'best_metric': self.state.best_metric,
        }
        torch.save(state_dict, checkpoint_dir / "trainer_state.pt")
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")