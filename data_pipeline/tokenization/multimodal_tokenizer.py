"""
多模态Tokenizer
支持图像、音频、视频的Token化处理
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import base64
from io import BytesIO

from .base_tokenizer import (
    BaseTokenizer,
    TokenizerConfig,
    TokenizedOutput,
    ModalityType,
)
from .text_tokenizer import TextTokenizer, TextTokenizerConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalTokenizerConfig(TokenizerConfig):
    """多模态Tokenizer配置"""
    # 文本配置
    text_tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # 图像配置
    image_processor_name: str = "openai/clip-vit-base-patch32"
    image_size: int = 224
    max_image_tokens: int = 256
    
    # 音频配置
    audio_processor_name: str = "openai/whisper-base"
    audio_sample_rate: int = 16000
    max_audio_tokens: int = 512
    
    # 视频配置
    video_frame_rate: int = 1  # 每秒提取帧数
    video_max_frames: int = 8
    max_video_tokens: int = 1024
    
    # 通用配置
    use_fast_processors: bool = True


class MultimodalTokenizer(BaseTokenizer):
    """
    多模态Tokenizer。
    支持文本、图像、音频、视频的统一Token化。
    """
    
    def __init__(self, config: MultimodalTokenizerConfig):
        super().__init__(config)
        self.multimodal_config = config
        
        # 子tokenizer
        self.text_tokenizer: Optional[TextTokenizer] = None
        
        # 处理器
        self.image_processor = None
        self.audio_processor = None
        self.video_processor = None
        
        # 模型
        self.image_encoder = None
        self.audio_encoder = None
        
        # 初始化
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        # 初始化文本tokenizer
        text_config = TextTokenizerConfig(
            tokenizer_name=self.multimodal_config.text_tokenizer_name,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
        )
        self.text_tokenizer = TextTokenizer(text_config)
        
        # 初始化图像处理器
        try:
            self._init_image_processor()
        except Exception as e:
            self.logger.warning(f"Failed to initialize image processor: {e}")
        
        # 初始化音频处理器
        try:
            self._init_audio_processor()
        except Exception as e:
            self.logger.warning(f"Failed to initialize audio processor: {e}")
        
        self.logger.info("Multimodal tokenizer initialized")
    
    def _init_image_processor(self):
        """初始化图像处理器"""
        # 实际实现：
        # from transformers import AutoImageProcessor, AutoModel
        # self.image_processor = AutoImageProcessor.from_pretrained(
        #     self.multimodal_config.image_processor_name
        # )
        # self.image_encoder = AutoModel.from_pretrained(
        #     self.multimodal_config.image_processor_name
        # )
        pass
    
    def _init_audio_processor(self):
        """初始化音频处理器"""
        # 实际实现：
        # from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        # self.audio_processor = AutoProcessor.from_pretrained(
        #     self.multimodal_config.audio_processor_name
        # )
        # self.audio_encoder = AutoModelForSpeechSeq2Seq.from_pretrained(
        #     self.multimodal_config.audio_processor_name
        # )
        pass
    
    def tokenize(self, data: Any, **kwargs) -> TokenizedOutput:
        """
        多模态Token化。
        
        Args:
            data: 输入数据（可以是文本、图像、音频、视频或多模态字典）
            
        Returns:
            TokenizedOutput: Token化输出
        """
        # 检测数据类型
        modality = self._detect_modality(data)
        
        # 根据模态选择处理方法
        if modality == ModalityType.TEXT:
            return self.tokenize_text(data, **kwargs)
        elif modality == ModalityType.IMAGE:
            return self.tokenize_image(data, **kwargs)
        elif modality == ModalityType.AUDIO:
            return self.tokenize_audio(data, **kwargs)
        elif modality == ModalityType.VIDEO:
            return self.tokenize_video(data, **kwargs)
        else:
            # 多模态
            return self.tokenize_multimodal(data, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """解码token IDs为文本"""
        if self.text_tokenizer:
            return self.text_tokenizer.decode(token_ids, **kwargs)
        return ""
    
    def tokenize_text(self, text: Union[str, Dict], **kwargs) -> TokenizedOutput:
        """文本Token化"""
        if self.text_tokenizer:
            return self.text_tokenizer.tokenize(text, **kwargs)
        
        # Fallback
        return TokenizedOutput(
            input_ids=[],
            attention_mask=[],
            modality=ModalityType.TEXT,
            metadata={'error': 'Text tokenizer not initialized'}
        )
    
    def tokenize_image(self, image: Any, **kwargs) -> TokenizedOutput:
        """
        图像Token化。
        
        Args:
            image: 图像数据（PIL Image、numpy array、路径或URL）
            
        Returns:
            TokenizedOutput: Token化输出
        """
        try:
            # 加载图像
            pil_image = self._load_image(image)
            
            if pil_image is None:
                return TokenizedOutput(
                    input_ids=[],
                    attention_mask=[],
                    modality=ModalityType.IMAGE,
                    metadata={'error': 'Failed to load image'}
                )
            
            # 处理图像
            if self.image_processor is not None:
                # 使用实际处理器
                # inputs = self.image_processor(images=pil_image, return_tensors="pt")
                # image_features = self.image_encoder(**inputs).last_hidden_state
                
                # 简化实现
                image_features = self._extract_image_features(pil_image)
            else:
                # 简化实现：生成占位符
                image_features = self._generate_image_tokens(pil_image)
            
            # 生成token序列
            num_tokens = min(self.multimodal_config.max_image_tokens, 256)
            image_tokens = [self._get_image_token_id()] * num_tokens
            
            return TokenizedOutput(
                input_ids=image_tokens,
                attention_mask=[1] * num_tokens,
                modality=ModalityType.IMAGE,
                image_features=image_features,
                metadata={
                    'image_size': pil_image.size,
                    'num_tokens': num_tokens,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error tokenizing image: {e}")
            return TokenizedOutput(
                input_ids=[],
                attention_mask=[],
                modality=ModalityType.IMAGE,
                metadata={'error': str(e)}
            )
    
    def tokenize_audio(self, audio: Any, **kwargs) -> TokenizedOutput:
        """
        音频Token化。
        
        Args:
            audio: 音频数据（文件路径、numpy array或音频对象）
            
        Returns:
            TokenizedOutput: Token化输出
        """
        try:
            # 加载音频
            audio_data = self._load_audio(audio)
            
            if audio_data is None:
                return TokenizedOutput(
                    input_ids=[],
                    attention_mask=[],
                    modality=ModalityType.AUDIO,
                    metadata={'error': 'Failed to load audio'}
                )
            
            # 处理音频
            if self.audio_processor is not None:
                # inputs = self.audio_processor(audio_data, return_tensors="pt", sampling_rate=16000)
                # audio_features = self.audio_encoder.encoder(**inputs).last_hidden_state
                
                # 简化实现
                audio_features = self._extract_audio_features(audio_data)
            else:
                audio_features = self._generate_audio_tokens(audio_data)
            
            # 生成token序列
            num_tokens = min(self.multimodal_config.max_audio_tokens, 512)
            audio_tokens = [self._get_audio_token_id()] * num_tokens
            
            return TokenizedOutput(
                input_ids=audio_tokens,
                attention_mask=[1] * num_tokens,
                modality=ModalityType.AUDIO,
                audio_features=audio_features,
                metadata={
                    'sample_rate': self.multimodal_config.audio_sample_rate,
                    'num_tokens': num_tokens,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error tokenizing audio: {e}")
            return TokenizedOutput(
                input_ids=[],
                attention_mask=[],
                modality=ModalityType.AUDIO,
                metadata={'error': str(e)}
            )
    
    def tokenize_video(self, video: Any, **kwargs) -> TokenizedOutput:
        """
        视频Token化。
        
        Args:
            video: 视频数据（文件路径或帧序列）
            
        Returns:
            TokenizedOutput: Token化输出
        """
        try:
            # 提取视频帧
            frames = self._extract_video_frames(video)
            
            if not frames:
                return TokenizedOutput(
                    input_ids=[],
                    attention_mask=[],
                    modality=ModalityType.VIDEO,
                    metadata={'error': 'Failed to extract video frames'}
                )
            
            # 对每一帧进行图像Token化
            frame_outputs = []
            for frame in frames:
                frame_output = self.tokenize_image(frame, **kwargs)
                frame_outputs.append(frame_output)
            
            # 合并所有帧的tokens
            video_tokens = []
            video_features = []
            
            for output in frame_outputs:
                video_tokens.extend(output.input_ids)
                if output.image_features is not None:
                    video_features.append(output.image_features)
            
            # 限制总token数
            max_tokens = self.multimodal_config.max_video_tokens
            if len(video_tokens) > max_tokens:
                video_tokens = video_tokens[:max_tokens]
            
            return TokenizedOutput(
                input_ids=video_tokens,
                attention_mask=[1] * len(video_tokens),
                modality=ModalityType.VIDEO,
                video_features=video_features if video_features else None,
                metadata={
                    'num_frames': len(frames),
                    'num_tokens': len(video_tokens),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error tokenizing video: {e}")
            return TokenizedOutput(
                input_ids=[],
                attention_mask=[],
                modality=ModalityType.VIDEO,
                metadata={'error': str(e)}
            )
    
    def tokenize_multimodal(self, data: Dict[str, Any], **kwargs) -> TokenizedOutput:
        """
        多模态数据Token化。
        
        Args:
            data: 多模态数据字典，包含text、image、audio等字段
            
        Returns:
            TokenizedOutput: 合并的Token化输出
        """
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        
        image_features = None
        audio_features = None
        video_features = None
        
        current_token_type = 0
        
        # 处理文本
        if 'text' in data:
            text_output = self.tokenize_text(data['text'], **kwargs)
            all_input_ids.extend(text_output.input_ids)
            all_attention_mask.extend(text_output.attention_mask)
            if text_output.token_type_ids:
                all_token_type_ids.extend(text_output.token_type_ids)
            else:
                all_token_type_ids.extend([current_token_type] * len(text_output.input_ids))
            current_token_type += 1
        
        # 处理图像
        if 'image' in data:
            image_output = self.tokenize_image(data['image'], **kwargs)
            all_input_ids.extend(image_output.input_ids)
            all_attention_mask.extend(image_output.attention_mask)
            all_token_type_ids.extend([current_token_type] * len(image_output.input_ids))
            image_features = image_output.image_features
            current_token_type += 1
        
        # 处理音频
        if 'audio' in data:
            audio_output = self.tokenize_audio(data['audio'], **kwargs)
            all_input_ids.extend(audio_output.input_ids)
            all_attention_mask.extend(audio_output.attention_mask)
            all_token_type_ids.extend([current_token_type] * len(audio_output.input_ids))
            audio_features = audio_output.audio_features
            current_token_type += 1
        
        # 处理视频
        if 'video' in data:
            video_output = self.tokenize_video(data['video'], **kwargs)
            all_input_ids.extend(video_output.input_ids)
            all_attention_mask.extend(video_output.attention_mask)
            all_token_type_ids.extend([current_token_type] * len(video_output.input_ids))
            video_features = video_output.video_features
        
        return TokenizedOutput(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            token_type_ids=all_token_type_ids if all_token_type_ids else None,
            modality=ModalityType.TEXT,  # 多模态标记为TEXT
            image_features=image_features,
            audio_features=audio_features,
            video_features=video_features,
            metadata={'multimodal': True}
        )
    
    def _detect_modality(self, data: Any) -> ModalityType:
        """检测数据模态"""
        if isinstance(data, str):
            return ModalityType.TEXT
        elif isinstance(data, dict):
            has_text = 'text' in data
            has_image = 'image' in data or 'image_path' in data or 'image_url' in data
            has_audio = 'audio' in data or 'audio_path' in data
            has_video = 'video' in data or 'video_path' in data
            
            # 多模态
            if sum([has_text, has_image, has_audio, has_video]) > 1:
                return ModalityType.TEXT  # 返回TEXT，实际会在tokenize_multimodal中处理
            
            if has_image:
                return ModalityType.IMAGE
            elif has_audio:
                return ModalityType.AUDIO
            elif has_video:
                return ModalityType.VIDEO
            elif has_text:
                return ModalityType.TEXT
        
        return ModalityType.TEXT
    
    def _load_image(self, image: Any):
        """加载图像"""
        try:
            from PIL import Image
            
            if isinstance(image, Image.Image):
                return image
            elif isinstance(image, str):
                if image.startswith('http'):
                    import requests
                    response = requests.get(image, timeout=10)
                    return Image.open(BytesIO(response.content))
                else:
                    return Image.open(image)
            elif isinstance(image, bytes):
                return Image.open(BytesIO(image))
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def _load_audio(self, audio: Any):
        """加载音频"""
        # 简化实现
        return audio
    
    def _extract_video_frames(self, video: Any) -> List:
        """提取视频帧"""
        # 简化实现
        return []
    
    def _extract_image_features(self, image):
        """提取图像特征"""
        # 简化实现：返回占位符
        return None
    
    def _extract_audio_features(self, audio):
        """提取音频特征"""
        return None
    
    def _generate_image_tokens(self, image):
        """生成图像tokens"""
        return None
    
    def _generate_audio_tokens(self, audio):
        """生成音频tokens"""
        return None
    
    def _get_image_token_id(self) -> int:
        """获取图像token ID"""
        if self.text_tokenizer and self.text_tokenizer.tokenizer:
            return self.text_tokenizer.get_token_id(self.config.image_token)
        return 0
    
    def _get_audio_token_id(self) -> int:
        """获取音频token ID"""
        if self.text_tokenizer and self.text_tokenizer.tokenizer:
            return self.text_tokenizer.get_token_id(self.config.audio_token)
        return 0