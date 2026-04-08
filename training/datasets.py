"""
训练数据集
支持预训练和SFT数据集
"""

from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str
    max_length: int = 2048
    image_resolution: int = 336
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42


class PretrainDataset(IterableDataset):
    """
    预训练数据集
    支持多模态数据混合采样
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        config: DataConfig,
        data_sources: Optional[Dict[str, float]] = None,
    ):
        """
        初始化预训练数据集
        
        Args:
            data_path: 数据路径
            tokenizer: tokenizer实例
            config: 数据配置
            data_sources: 数据源及其权重，如{"text": 0.5, "image_text": 0.3, "audio": 0.2}
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.data_sources = data_sources or {"text": 1.0}
        
        self.logger = logging.getLogger(__name__)
        self._load_data_index()
    
    def _load_data_index(self):
        """加载数据索引"""
        index_file = self.data_path / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.data_index = json.load(f)
        else:
            # 扫描数据文件
            self.data_index = self._scan_data_files()
            self.logger.info(f"Found {len(self.data_index)} data files")
    
    def _scan_data_files(self) -> List[Dict[str, Any]]:
        """扫描数据文件"""
        data_files = []
        for file_path in self.data_path.rglob("*.jsonl"):
            data_files.append({
                "path": str(file_path),
                "type": self._infer_data_type(file_path),
            })
        return data_files
    
    def _infer_data_type(self, file_path: Path) -> str:
        """推断数据类型"""
        path_str = str(file_path).lower()
        if "image" in path_str or "img" in path_str:
            return "image_text"
        elif "audio" in path_str:
            return "audio"
        elif "video" in path_str:
            return "video"
        else:
            return "text"
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代数据"""
        for data_file in self.data_index:
            file_path = Path(data_file["path"])
            data_type = data_file["type"]
            
            # 读取JSONL文件
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    processed = self._process_sample(sample, data_type)
                    if processed is not None:
                        yield processed
    
    def _process_sample(self, sample: Dict[str, Any], data_type: str) -> Optional[Dict[str, torch.Tensor]]:
        """处理单个样本"""
        try:
            if data_type == "text":
                return self._process_text(sample)
            elif data_type == "image_text":
                return self._process_image_text(sample)
            elif data_type == "audio":
                return self._process_audio(sample)
            elif data_type == "video":
                return self._process_video(sample)
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Error processing sample: {e}")
            return None
    
    def _process_text(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理文本样本"""
        text = sample.get("text", "")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }
    
    def _process_image_text(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理图文对样本"""
        text = sample.get("text", "")
        image_path = sample.get("image_path")
        
        # 加载图像
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            # 图像预处理会在collator中完成
            pixel_values = image
        else:
            pixel_values = None
        
        # 构建输入文本
        prompt = f"<image>\n{text}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
            "pixel_values": pixel_values,
        }
    
    def _process_audio(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理音频样本"""
        # TODO: 实现音频处理
        return None
    
    def _process_video(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理视频样本"""
        # TODO: 实现视频处理
        return None


class SFTDataset(Dataset):
    """
    监督微调数据集
    支持多轮对话格式
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        config: DataConfig,
        template: str = "chatml",
    ):
        """
        初始化SFT数据集
        
        Args:
            data_path: 数据路径
            tokenizer: tokenizer实例
            config: 数据配置
            template: 对话模板(chatml/llama3/vicuna等)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.template = template
        
        self.logger = logging.getLogger(__name__)
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        self.data = []
        
        # 支持JSONL和JSON格式
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        
        self.logger.info(f"Loaded {len(self.data)} SFT samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.data[idx]
        return self._process_sample(sample)
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理单个样本"""
        # 提取对话
        conversations = sample.get("conversations", [])
        if not conversations:
            conversations = sample.get("messages", [])
        
        # 应用对话模板
        prompt = self._apply_template(conversations)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 构建labels（mask指令部分）
        labels = self._build_labels(conversations, input_ids)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # 处理图像
        if "image" in sample or "images" in sample:
            image_path = sample.get("image") or sample.get("images", [None])[0]
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                result["pixel_values"] = image
        
        return result
    
    def _apply_template(self, conversations: List[Dict[str, str]]) -> str:
        """应用对话模板"""
        if self.template == "chatml":
            return self._apply_chatml_template(conversations)
        elif self.template == "llama3":
            return self._apply_llama3_template(conversations)
        else:
            return self._apply_default_template(conversations)
    
    def _apply_chatml_template(self, conversations: List[Dict[str, str]]) -> str:
        """应用ChatML模板"""
        prompt_parts = []
        for conv in conversations:
            role = conv.get("role", "user")
            content = conv.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        # 添加assistant开始标记
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "".join(prompt_parts)
    
    def _apply_llama3_template(self, conversations: List[Dict[str, str]]) -> str:
        """应用Llama3模板"""
        prompt_parts = ["<|begin_of_text|>"]
        
        for conv in conversations:
            role = conv.get("role", "user")
            content = conv.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "user":
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant":
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        # 添加assistant开始标记
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(prompt_parts)
    
    def _apply_default_template(self, conversations: List[Dict[str, str]]) -> str:
        """应用默认模板"""
        prompt_parts = []
        for conv in conversations:
            role = conv.get("role", "user")
            content = conv.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")
        
        return "".join(prompt_parts)
    
    def _build_labels(self, conversations: List[Dict[str, str]], input_ids: torch.Tensor) -> torch.Tensor:
        """构建labels，mask指令部分"""
        labels = input_ids.clone()
        
        # 找到assistant回复的位置
        # 这里简化处理，实际需要根据具体模板精确mask
        # 通常将user和system部分的token设为-100
        
        # 解码input_ids以找到assistant标记
        text = self.tokenizer.decode(input_ids)
        
        # 找到所有assistant回复的开始和结束位置
        # 这里使用简单的字符串匹配，实际应该更精确
        assistant_start = text.find("<|im_start|>assistant")
        if assistant_start == -1:
            assistant_start = text.find("Assistant:")
        
        if assistant_start != -1:
            # 找到对应的token位置
            # 简化处理：假设前半部分是指令，后半部分是回复
            # 实际应该精确计算
            pass
        
        return labels


class PreferenceDataset(Dataset):
    """
    偏好数据集（用于DPO/RLHF）
    包含prompt、chosen response和rejected response
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        config: DataConfig,
    ):
        """
        初始化偏好数据集
        
        Args:
            data_path: 数据路径
            tokenizer: tokenizer实例
            config: 数据配置
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        
        self.logger = logging.getLogger(__name__)
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        self.data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(self.data)} preference samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.data[idx]
        
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        # 构建chosen和rejected的完整输入
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        # Tokenize
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
        }