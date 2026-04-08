"""
监督微调训练器（Supervised Fine-Tuning）
支持全参数微调和LoRA/QLoRA高效微调
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .base_trainer import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(TrainingConfig):
    """SFT训练配置"""
    # LoRA配置
    use_lora: bool = False
    lora_r: int = 8  # LoRA秩
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  # 默认: q_proj, k_proj, v_proj, o_proj
    
    # QLoRA配置
    use_qlora: bool = False
    qlora_bits: int = 4  # 量化位数
    
    # 训练配置
    max_length: int = 2048
    mask_instruction: bool = True  # 是否mask指令部分的loss
    packing: bool = False  # 是否打包多个样本
    
    # 学习率
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                        "gate_proj", "up_proj", "down_proj"]


class SFTTrainer(BaseTrainer):
    """
    监督微调训练器
    支持全参数微调和LoRA/QLoRA高效微调
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # 应用LoRA/QLoRA
        if config.use_lora or config.use_qlora:
            model = self._apply_peft(model)
        
        super().__init__(model, config, train_dataset, eval_dataset)
        self.logger = logging.getLogger(__name__)
    
    def _apply_peft(self, model: nn.Module) -> nn.Module:
        """应用参数高效微调（LoRA/QLoRA）"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError("请安装peft库: pip install peft")
        
        if self.config.use_qlora:
            # QLoRA: 4-bit量化 + LoRA
            model = prepare_model_for_kbit_training(model)
            self.logger.info("Applied QLoRA (4-bit quantization + LoRA)")
        
        # LoRA配置
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        self.logger.info(f"Applied LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        return model
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步训练"""
        # 前向传播
        outputs = self.model(**batch)
        
        # 计算loss（仅计算assistant回复部分）
        loss = outputs.loss
        
        return loss
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算交叉熵损失
        仅计算assistant回复部分的loss（mask指令部分）
        """
        # labels中-100表示不计算loss的部分
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 展平
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def train(self):
        """执行SFT训练"""
        self.logger.info("Starting SFT training...")
        
        # 设置数据加载器
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置学习率调度器
        num_training_steps = len(train_dataloader) * self.config.num_train_epochs
        self.lr_scheduler = self._setup_lr_scheduler(num_training_steps)
        
        # 训练循环
        for epoch in range(self.config.num_train_epochs):
            self.state.epoch = epoch
            self._train_epoch(train_dataloader)
            
            # 评估
            if self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                self.logger.info(f"Epoch {epoch} evaluation: {eval_metrics}")
                
                # 保存最佳模型
                if eval_metrics.get('eval_loss', float('inf')) < self.state.best_metric:
                    self.state.best_metric = eval_metrics['eval_loss']
                    self.save_model(str(Path(self.config.output_dir) / "best_model"))
            
            # 保存checkpoint
            self.save_checkpoint(epoch)
        
        self.logger.info("SFT training completed")
        return self.state
    
    def _train_epoch(self, dataloader: DataLoader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.config.fp16 or self.config.bf16):
                loss = self._training_step(batch)
            
            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            # 更新参数
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 优化器步进
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 学习率调度
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                self.state.global_step += 1
                
                # 日志记录
                if self.state.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    self.logger.info(
                        f"Epoch {self.state.epoch} Step {self.state.global_step}: "
                        f"loss={avg_loss:.4f}, lr={self.lr_scheduler.get_last_lr()[0]:.2e}"
                    )
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 区分LoRA参数和其他参数
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'lora' in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': self.config.learning_rate})
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.config.learning_rate})
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )
        
        return optimizer
    
    def _setup_lr_scheduler(self, num_training_steps: int):
        """设置学习率调度器"""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return scheduler
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """批处理函数"""
        # 收集所有键
        keys = batch[0].keys()
        batch_dict = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if isinstance(values[0], torch.Tensor):
                # 动态padding
                if key in ['input_ids', 'labels', 'attention_mask']:
                    batch_dict[key] = self._pad_sequence(values, key)
                else:
                    batch_dict[key] = torch.stack(values)
            else:
                batch_dict[key] = values
        
        return batch_dict
    
    def _pad_sequence(self, sequences: List[torch.Tensor], key: str) -> torch.Tensor:
        """动态padding"""
        max_len = max(seq.size(0) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if key == 'input_ids':
                pad_value = self.tokenizer.pad_token_id or 0
            elif key == 'labels':
                pad_value = -100  # 不计算loss
            elif key == 'attention_mask':
                pad_value = 0
            else:
                pad_value = 0
            
            if seq.size(0) < max_len:
                padding = torch.full((max_len - seq.size(0),), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            
            padded.append(padded_seq)
        
        return torch.stack(padded)
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
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
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # 保存tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
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