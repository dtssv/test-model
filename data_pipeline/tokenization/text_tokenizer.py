"""
文本Tokenizer
基于HuggingFace Tokenizers的文本Token化处理
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

from .base_tokenizer import (
    BaseTokenizer,
    TokenizerConfig,
    TokenizedOutput,
    ModalityType,
)

logger = logging.getLogger(__name__)


@dataclass
class TextTokenizerConfig(TokenizerConfig):
    """文本Tokenizer配置"""
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_fast: bool = True
    
    # 编码选项
    add_bos_token: bool = True
    add_eos_token: bool = True
    
    # 特殊处理
    clean_up_tokenization_spaces: bool = True
    use_default_system_prompt: bool = False
    
    # Chat模板
    chat_template: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # 并行处理
    num_threads: int = 4


class TextTokenizer(BaseTokenizer):
    """
    文本Tokenizer。
    支持多种预训练模型的文本Token化。
    """
    
    def __init__(self, config: TextTokenizerConfig):
        super().__init__(config)
        self.text_config = config
        
        # 初始化tokenizer
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """初始化HuggingFace Tokenizer"""
        try:
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.text_config.tokenizer_name,
                use_fast=self.text_config.use_fast,
                trust_remote_code=True,
            )
            
            # 更新词汇表
            self.vocab = self.tokenizer.get_vocab()
            
            # 更新特殊token
            self.special_tokens = {
                'bos_token': self.tokenizer.bos_token,
                'eos_token': self.tokenizer.eos_token,
                'pad_token': self.tokenizer.pad_token,
                'unk_token': self.tokenizer.unk_token,
                'sep_token': self.tokenizer.sep_token,
                'cls_token': self.tokenizer.cls_token,
                'mask_token': self.tokenizer.mask_token,
            }
            
            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Text tokenizer initialized: {self.text_config.tokenizer_name}")
            self.logger.info(f"Vocab size: {len(self.vocab)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            raise
    
    def tokenize(
        self,
        text: Union[str, Dict[str, Any]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[bool] = None,
        return_tokens: bool = False,
        return_offsets: bool = False,
        **kwargs
    ) -> TokenizedOutput:
        """
        文本Token化。
        
        Args:
            text: 输入文本或文本字典
            add_special_tokens: 是否添加特殊token
            max_length: 最大长度
            truncation: 是否截断
            padding: 是否填充
            return_tokens: 是否返回tokens
            return_offsets: 是否返回偏移量
            
        Returns:
            TokenizedOutput: Token化输出
        """
        # 提取文本
        if isinstance(text, dict):
            text_content = text.get('text', '')
        else:
            text_content = text
        
        if not text_content:
            return TokenizedOutput(
                input_ids=[],
                attention_mask=[],
                modality=ModalityType.TEXT,
                metadata={'error': 'Empty text'}
            )
        
        # 检查缓存
        cache_key = self._get_cache_key(text_content)
        if cache_key:
            cached = self._get_cached_output(cache_key)
            if cached:
                return cached
        
        # 设置参数
        max_length = max_length or self.config.max_length
        truncation = truncation if truncation is not None else self.config.truncation
        padding = padding if padding is not None else self.config.padding
        
        try:
            # 使用tokenizer编码
            encoded = self.tokenizer(
                text_content,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=False,  # 先不填充，后面统一处理
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors=None,  # 返回列表
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            token_type_ids = encoded.get('token_type_ids')
            
            # 填充
            is_padded = False
            if padding and len(input_ids) < max_length:
                input_ids, attention_mask = self._pad(
                    input_ids,
                    max_length,
                    attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                if token_type_ids is not None:
                    token_type_ids = token_type_ids + [0] * (max_length - len(token_type_ids))
                is_padded = True
            
            # 获取tokens
            tokens = None
            if return_tokens:
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # 获取偏移量
            offsets = None
            if return_offsets:
                offsets = self._get_offsets(text_content, input_ids)
            
            # 创建输出
            output = TokenizedOutput(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                modality=ModalityType.TEXT,
                tokens=tokens,
                offsets=offsets,
                metadata={
                    'tokenizer': self.text_config.tokenizer_name,
                    'original_length': len(encoded['input_ids']),
                    'truncated': len(encoded['input_ids']) > max_length if truncation else False,
                    'padded': is_padded,
                }
            )
            
            # 更新统计
            self.stats.update(
                output,
                truncated=len(encoded['input_ids']) > max_length if truncation else False,
                padded=is_padded
            )
            
            # 缓存
            if cache_key:
                self._cache_output(cache_key, output)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            return TokenizedOutput(
                input_ids=[],
                attention_mask=[],
                modality=ModalityType.TEXT,
                metadata={'error': str(e)}
            )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        解码token IDs为文本。
        
        Args:
            token_ids: token IDs
            skip_special_tokens: 是否跳过特殊token
            clean_up_tokenization_spaces: 是否清理tokenization空格
            
        Returns:
            str: 解码后的文本
        """
        if not token_ids:
            return ""
        
        clean_up = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None else self.text_config.clean_up_tokenization_spaces
        
        try:
            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up,
            )
            return text
        except Exception as e:
            self.logger.error(f"Error decoding tokens: {e}")
            return ""
    
    def tokenize_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs
    ) -> TokenizedOutput:
        """
        Chat对话Token化。
        
        Args:
            messages: 对话消息列表
            add_generation_prompt: 是否添加生成提示
            
        Returns:
            TokenizedOutput: Token化输出
        """
        try:
            # 使用tokenizer的chat模板
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            else:
                # 手动构建对话格式
                text = self._build_chat_text(messages, add_generation_prompt)
            
            return self.tokenize(text, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error tokenizing chat: {e}")
            return TokenizedOutput(
                input_ids=[],
                attention_mask=[],
                modality=ModalityType.TEXT,
                metadata={'error': str(e)}
            )
    
    def _build_chat_text(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> str:
        """
        手动构建对话文本。
        
        Args:
            messages: 对话消息列表
            add_generation_prompt: 是否添加生成提示
            
        Returns:
            str: 构建的对话文本
        """
        text_parts = []
        
        # 添加系统提示
        if self.text_config.system_prompt:
            text_parts.append(f"<|system|>\n{self.text_config.system_prompt}\n")
        
        # 添加对话消息
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                text_parts.append(f"<|system|>\n{content}\n")
            elif role == 'user':
                text_parts.append(f"<|user|>\n{content}\n")
            elif role == 'assistant':
                text_parts.append(f"<|assistant|>\n{content}\n")
        
        # 添加生成提示
        if add_generation_prompt:
            text_parts.append("<|assistant|>\n")
        
        return ''.join(text_parts)
    
    def _get_offsets(self, text: str, token_ids: List[int]) -> List[tuple]:
        """
        获取token偏移量。
        
        Args:
            text: 原始文本
            token_ids: token IDs
            
        Returns:
            List[tuple]: 偏移量列表
        """
        try:
            # 使用tokenizer的encode_plus获取偏移量
            encoded = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            return encoded.get('offset_mapping', [])
        except Exception:
            return []
    
    def get_token_id(self, token: str) -> int:
        """获取token的ID"""
        return self.vocab.get(token, self.vocab.get(self.config.unk_token, 0))
    
    def get_token(self, token_id: int) -> str:
        """获取ID对应的token"""
        try:
            return self.tokenizer.convert_ids_to_tokens([token_id])[0]
        except Exception:
            return self.config.unk_token
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将tokens转换为字符串"""
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def num_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        截断文本到指定token数量。
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            
        Returns:
            str: 截断后的文本
        """
        tokens = self.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.decode(truncated_tokens)