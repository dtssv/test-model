"""
vLLM推理引擎
使用vLLM实现高效推理
"""

from typing import Optional, List, Dict, Any, Generator
import logging
import torch

from .base_engine import InferenceEngine, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


class VLLMEngine(InferenceEngine):
    """
    vLLM推理引擎
    支持PagedAttention和连续批处理
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        **kwargs
    ):
        """
        初始化vLLM引擎
        
        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行度
            gpu_memory_utilization: GPU显存利用率
            max_model_len: 最大模型长度
            **kwargs: 其他参数
        """
        super().__init__(model_path, **kwargs)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None
    
    def load_model(self):
        """加载模型"""
        try:
            from vllm import LLM, SamplingParams
            self.logger.info(f"Loading model with vLLM from {self.model_path}")
            
            # 初始化vLLM引擎
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
            )
            
            # 获取tokenizer
            self.tokenizer = self.llm.get_tokenizer()
            
            self.logger.info("vLLM engine loaded successfully")
            
        except ImportError:
            raise ImportError(
                "vLLM未安装。请安装: pip install vllm"
            )
    
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
        if self.llm is None:
            self.load_model()
        
        from vllm import SamplingParams
        
        # 构建采样参数
        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            length_penalty=config.length_penalty,
            early_stopping=config.early_stopping,
            stop=config.stop_tokens,
            stop_token_ids=config.stop_token_ids,
        )
        
        # 生成
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # 提取结果
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        finish_reason = output.outputs[0].finish_reason
        
        # 统计token数量
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(token_ids)
        
        return GenerationResult(
            text=generated_text,
            token_ids=list(token_ids),
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
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
        if self.llm is None:
            self.load_model()
        
        # vLLM支持流式输出
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop_tokens,
            stop_token_ids=config.stop_token_ids,
        )
        
        # 流式生成
        for output in self.llm.generate_stream([prompt], sampling_params):
            if output.outputs:
                yield output.outputs[0].text
    
    def encode(self, text: str) -> List[int]:
        """
        编码文本为token IDs
        
        Args:
            text: 输入文本
        
        Returns:
            token IDs列表
        """
        if self.tokenizer is None:
            if self.llm is None:
                self.load_model()
        
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        解码token IDs为文本
        
        Args:
            token_ids: token IDs列表
        
        Returns:
            解码后的文本
        """
        if self.tokenizer is None:
            if self.llm is None:
                self.load_model()
        
        return self.tokenizer.decode(token_ids)
    
    def batch_generate(
        self,
        prompts: List[str],
        config: GenerationConfig,
        **kwargs
    ) -> List[GenerationResult]:
        """
        批量生成文本（vLLM优化）
        
        Args:
            prompts: 输入提示列表
            config: 生成配置
            **kwargs: 其他参数
        
        Returns:
            生成结果列表
        """
        if self.llm is None:
            self.load_model()
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop_tokens,
            stop_token_ids=config.stop_token_ids,
        )
        
        # 批量生成
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(token_ids)
            
            results.append(GenerationResult(
                text=generated_text,
                token_ids=list(token_ids),
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ))
        
        return results
    
    def get_model_memory_footprint(self) -> Dict[str, float]:
        """
        获取模型显存占用
        
        Returns:
            显存占用信息（MB）
        """
        if not torch.cuda.is_available():
            return {"gpu_memory_mb": 0}
        
        # vLLM管理自己的内存，这里返回GPU总使用情况
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
        }
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = super().model_info
        info.update({
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "backend": "vLLM",
        })
        return info