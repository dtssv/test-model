"""
推理引擎基类
定义推理引擎的标准接口
"""

from typing import Optional, List, Dict, Any, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    num_return_sequences: int = 1
    do_sample: bool = True
    use_beam_search: bool = False
    beam_width: int = 1
    
    # 停止条件
    stop_tokens: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    
    # 流式输出
    stream: bool = False


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    token_ids: List[int]
    finish_reason: str
    usage: Dict[str, int]
    
    # 元数据
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class InferenceEngine(ABC):
    """
    推理引擎基类
    定义所有推理引擎的标准接口
    """
    
    def __init__(self, model_path: str, **kwargs):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.model_path = model_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            config: 生成配置
            **kwargs: 其他参数
        
        Returns:
            生成结果
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示
            config: 生成配置
            **kwargs: 其他参数
        
        Yields:
            生成的文本片段
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        编码文本为token IDs
        
        Args:
            text: 输入文本
        
        Returns:
            token IDs列表
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        解码token IDs为文本
        
        Args:
            token_ids: token IDs列表
        
        Returns:
            解码后的文本
        """
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        config: GenerationConfig,
        **kwargs
    ) -> List[GenerationResult]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            config: 生成配置
            **kwargs: 其他参数
        
        Returns:
            生成结果列表
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config, **kwargs)
            results.append(result)
        return results
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        **kwargs
    ) -> GenerationResult:
        """
        多轮对话生成
        
        Args:
            messages: 对话消息列表，格式为[{"role": "user", "content": "..."}]
            config: 生成配置
            **kwargs: 其他参数
        
        Returns:
            生成结果
        """
        # 将对话消息转换为单个prompt
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, config, **kwargs)
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        多轮对话流式生成
        
        Args:
            messages: 对话消息列表
            config: 生成配置
            **kwargs: 其他参数
        
        Yields:
            生成的文本片段
        """
        prompt = self._messages_to_prompt(messages)
        yield from self.generate_stream(prompt, config, **kwargs)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        将对话消息转换为prompt
        
        Args:
            messages: 对话消息列表
        
        Returns:
            格式化的prompt
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n</|system|>\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n</|user|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n</|assistant|>\n")
        
        # 添加assistant开始标记
        prompt_parts.append("<|assistant|>\n")
        
        return "".join(prompt_parts)
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_path": self.model_path,
            "model_type": self.__class__.__name__,
        }
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        获取tokenizer信息
        
        Returns:
            tokenizer信息字典
        """
        if self.tokenizer is None:
            return {}
        
        return {
            "vocab_size": getattr(self.tokenizer, "vocab_size", None),
            "model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "padding_side": getattr(self.tokenizer, "padding_side", None),
        }
    
    @abstractmethod
    def get_model_memory_footprint(self) -> Dict[str, float]:
        """
        获取模型显存占用
        
        Returns:
            显存占用信息（MB）
        """
        pass
    
    def release_memory(self):
        """释放模型占用的显存"""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU memory released")