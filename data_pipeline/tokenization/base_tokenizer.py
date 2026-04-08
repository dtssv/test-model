"""
Tokenizer基类
定义多模态数据Token化的标准接口
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class TokenizerConfig:
    """Tokenizer配置基类"""
    # 基本配置
    max_length: int = 2048
    truncation: bool = True
    padding: bool = True
    return_tensors: str = "pt"  # pt, np, list
    
    # 特殊token
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    sep_token: str = "<sep>"
    cls_token: str = "<cls>"
    mask_token: str = "<mask>"
    
    # 模态特定
    image_token: str = "<image>"
    audio_token: str = "<audio>"
    video_token: str = "<video>"
    
    # 缓存
    enable_cache: bool = True
    cache_dir: Optional[str] = None


@dataclass
class TokenizedOutput:
    """Token化输出"""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    modality: ModalityType = ModalityType.TEXT
    
    # 多模态特定
    image_features: Optional[Any] = None
    audio_features: Optional[Any] = None
    video_features: Optional[Any] = None
    
    # 元数据
    tokens: Optional[List[str]] = None
    offsets: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'modality': self.modality.value,
        }
        
        if self.token_type_ids is not None:
            result['token_type_ids'] = self.token_type_ids
        
        if self.image_features is not None:
            result['image_features'] = self.image_features
        
        if self.audio_features is not None:
            result['audio_features'] = self.audio_features
        
        if self.video_features is not None:
            result['video_features'] = self.video_features
        
        if self.tokens is not None:
            result['tokens'] = self.tokens
        
        if self.offsets is not None:
            result['offsets'] = self.offsets
        
        result['metadata'] = self.metadata
        
        return result


@dataclass
class TokenizationStats:
    """Token化统计信息"""
    total_items: int = 0
    total_tokens: int = 0
    avg_tokens_per_item: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    truncated_items: int = 0
    padded_items: int = 0
    errors: int = 0
    
    def update(self, output: TokenizedOutput, truncated: bool = False, padded: bool = False):
        """更新统计"""
        self.total_items += 1
        num_tokens = len(output.input_ids)
        self.total_tokens += num_tokens
        self.max_tokens = max(self.max_tokens, num_tokens)
        self.min_tokens = min(self.min_tokens, num_tokens) if self.min_tokens > 0 else num_tokens
        
        if truncated:
            self.truncated_items += 1
        
        if padded:
            self.padded_items += 1
        
        self.avg_tokens_per_item = self.total_tokens / self.total_items
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_items': self.total_items,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_item': self.avg_tokens_per_item,
            'max_tokens': self.max_tokens,
            'min_tokens': self.min_tokens,
            'truncated_items': self.truncated_items,
            'padded_items': self.padded_items,
            'errors': self.errors,
        }


class BaseTokenizer(ABC):
    """
    Tokenizer抽象基类。
    定义所有Tokenizer的标准接口。
    """
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stats = TokenizationStats()
        
        # Tokenizer实例
        self.tokenizer = None
        self.vocab = {}
        self.special_tokens = {}
        
        # 缓存
        self._cache: Dict[str, TokenizedOutput] = {}
    
    @abstractmethod
    def tokenize(self, data: Any, **kwargs) -> TokenizedOutput:
        """
        执行Token化。
        
        Args:
            data: 输入数据
            
        Returns:
            TokenizedOutput: Token化输出
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        将token ID解码为文本。
        
        Args:
            token_ids: token ID列表
            
        Returns:
            str: 解码后的文本
        """
        pass
    
    def tokenize_batch(self, data_list: List[Any], **kwargs) -> List[TokenizedOutput]:
        """
        批量Token化。
        
        Args:
            data_list: 数据列表
            
        Returns:
            List[TokenizedOutput]: Token化输出列表
        """
        outputs = []
        
        for data in data_list:
            try:
                output = self.tokenize(data, **kwargs)
                outputs.append(output)
                self.stats.update(output)
            except Exception as e:
                self.logger.error(f"Error tokenizing data: {e}")
                self.stats.errors += 1
        
        return outputs
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        编码文本为token IDs。
        
        Args:
            text: 输入文本
            
        Returns:
            List[int]: token IDs
        """
        output = self.tokenize(text, **kwargs)
        return output.input_ids
    
    def encode_plus(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[bool] = None,
        **kwargs
    ) -> TokenizedOutput:
        """
        增强编码，返回完整输出。
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
            max_length: 最大长度
            truncation: 是否截断
            padding: 是否填充
            
        Returns:
            TokenizedOutput: 完整输出
        """
        max_length = max_length or self.config.max_length
        truncation = truncation if truncation is not None else self.config.truncation
        padding = padding if padding is not None else self.config.padding
        
        return self.tokenize(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            **kwargs
        )
    
    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表"""
        return self.vocab
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab)
    
    def get_special_tokens(self) -> Dict[str, str]:
        """获取特殊token"""
        return self.special_tokens
    
    def add_special_tokens(self, special_tokens: Dict[str, str]):
        """
        添加特殊token。
        
        Args:
            special_tokens: 特殊token字典
        """
        self.special_tokens.update(special_tokens)
        # 更新tokenizer
        if self.tokenizer is not None:
            self.tokenizer.add_special_tokens(special_tokens)
    
    def get_stats(self) -> TokenizationStats:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = TokenizationStats()
    
    def _truncate(
        self,
        input_ids: List[int],
        max_length: int,
        attention_mask: Optional[List[int]] = None
    ) -> Tuple[List[int], Optional[List[int]]]:
        """
        截断序列。
        
        Args:
            input_ids: token IDs
            max_length: 最大长度
            attention_mask: 注意力掩码
            
        Returns:
            Tuple: 截断后的序列
        """
        if len(input_ids) <= max_length:
            return input_ids, attention_mask
        
        truncated_ids = input_ids[:max_length]
        truncated_mask = attention_mask[:max_length] if attention_mask else None
        
        return truncated_ids, truncated_mask
    
    def _pad(
        self,
        input_ids: List[int],
        max_length: int,
        attention_mask: Optional[List[int]] = None,
        pad_token_id: int = 0
    ) -> Tuple[List[int], Optional[List[int]]]:
        """
        填充序列。
        
        Args:
            input_ids: token IDs
            max_length: 最大长度
            attention_mask: 注意力掩码
            pad_token_id: 填充token ID
            
        Returns:
            Tuple: 填充后的序列
        """
        padding_length = max_length - len(input_ids)
        
        if padding_length <= 0:
            return input_ids, attention_mask
        
        padded_ids = input_ids + [pad_token_id] * padding_length
        
        if attention_mask is not None:
            padded_mask = attention_mask + [0] * padding_length
        else:
            padded_mask = [1] * len(input_ids) + [0] * padding_length
        
        return padded_ids, padded_mask
    
    def _add_special_tokens_to_sequence(
        self,
        input_ids: List[int],
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> List[int]:
        """
        添加特殊token到序列。
        
        Args:
            input_ids: token IDs
            bos_token_id: 开始token ID
            eos_token_id: 结束token ID
            
        Returns:
            List[int]: 添加特殊token后的序列
        """
        result = []
        
        if bos_token_id is not None:
            result.append(bos_token_id)
        
        result.extend(input_ids)
        
        if eos_token_id is not None:
            result.append(eos_token_id)
        
        return result
    
    def _get_cache_key(self, data: Any) -> Optional[str]:
        """生成缓存键"""
        if isinstance(data, str):
            return f"text_{hash(data)}"
        elif isinstance(data, dict) and 'text' in data:
            return f"text_{hash(data['text'])}"
        else:
            return None
    
    def _get_cached_output(self, key: str) -> Optional[TokenizedOutput]:
        """从缓存获取输出"""
        if self.config.enable_cache and key in self._cache:
            return self._cache[key]
        return None
    
    def _cache_output(self, key: str, output: TokenizedOutput):
        """缓存输出"""
        if self.config.enable_cache:
            self._cache[key] = output
    
    def save_pretrained(self, save_dir: str):
        """
        保存tokenizer到目录。
        
        Args:
            save_dir: 保存目录
        """
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)
            self.logger.info(f"Tokenizer saved to {save_dir}")
    
    def load_pretrained(self, load_dir: str):
        """
        从目录加载tokenizer。
        
        Args:
            load_dir: 加载目录
        """
        # 子类实现
        pass
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return self.get_vocab_size()
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(vocab_size={self.get_vocab_size()})"