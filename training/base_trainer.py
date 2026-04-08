"""
训练器基类
定义模型训练的标准接口
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


@dataclass
class TrainingState:
    """训练状态"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    loss_history: list = field(default_factory=list)


class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = TrainingState()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup()
    
    def _setup(self):
        """初始化"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(self.config.seed)
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        self.logger.info("Trainer initialized")
    
    def train(self) -> TrainingState:
        """执行训练"""
        self.logger.info("Starting training...")
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
        )
        
        for epoch in range(self.config.num_train_epochs):
            self.state.epoch = epoch
            self._train_epoch(dataloader)
        
        self.logger.info("Training completed")
        return self.state
    
    def _train_epoch(self, dataloader: DataLoader):
        """训练一个epoch"""
        self.model.train()
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                loss = self._training_step(batch)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.state.global_step += 1
            
            if self.state.global_step % self.config.logging_steps == 0:
                self.logger.info(f"Step {self.state.global_step}: loss={loss.item():.4f}")
    
    @abstractmethod
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """单步训练"""
        pass
    
    def save_model(self, output_dir: str):
        """保存模型"""
        torch.save(self.model.state_dict(), Path(output_dir) / "model.pt")
        self.logger.info(f"Model saved to {output_dir}")