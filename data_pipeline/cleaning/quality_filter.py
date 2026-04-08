"""
数据质量过滤器
多维度数据质量评分和过滤
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .base_cleaner import BaseCleaner, CleaningConfig, CleanedDataItem, CleaningStatus

logger = logging.getLogger(__name__)


@dataclass
class QualityFilterConfig(CleaningConfig):
    """质量过滤器配置"""
    min_text_quality: float = 0.5
    min_image_quality: float = 0.5
    min_audio_quality: float = 0.5
    min_alignment_score: float = 0.3
    enable_multimodal_scoring: bool = True
    # 文本质量维度
    text_fluency_weight: float = 0.3
    text_information_density_weight: float = 0.3
    text_education_value_weight: float = 0.4
    # 图像质量维度
    image_aesthetic_weight: float = 0.4
    image_clarity_weight: float = 0.3
    image_nsfw_weight: float = 0.3
    # 音频质量维度
    audio_snr_weight: float = 0.4
    audio_clarity_weight: float = 0.3
    audio_speaker_quality_weight: float = 0.3


@dataclass
class TextQualityScore:
    """文本质量评分"""
    fluency: float  # 流畅度
    information_density: float  # 信息密度
    education_value: float  # 教育价值
    composite: float  # 综合评分
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'fluency': self.fluency,
            'information_density': self.information_density,
            'education_value': self.education_value,
            'composite': self.composite
        }


@dataclass
class ImageQualityScore:
    """图像质量评分"""
    aesthetic: float  # 美学评分
    clarity: float  # 清晰度
    nsfw_score: float  # NSFW概率（越低越好）
    composite: float  # 综合评分
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'aesthetic': self.aesthetic,
            'clarity': self.clarity,
            'nsfw_score': self.nsfw_score,
            'composite': self.composite
        }


@dataclass
class AudioQualityScore:
    """音频质量评分"""
    snr: float  # 信噪比
    clarity: float  # 语音清晰度
    speaker_quality: float  # 说话人质量
    composite: float  # 综合评分
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'snr': self.snr,
            'clarity': self.clarity,
            'speaker_quality': self.speaker_quality,
            'composite': self.composite
        }


class QualityFilter(BaseCleaner):
    """
    多维度数据质量评分器。
    文本维度：流畅度、信息密度、教育价值。
    图像维度：美学评分、清晰度、NSFW概率。
    音频维度：信噪比、语音清晰度、说话人质量。
    """
    
    def __init__(self, config: QualityFilterConfig):
        super().__init__(config)
        self.quality_config = config
        
        # 质量评估模型
        self.text_quality_model = None
        self.image_quality_model = None
        self.audio_quality_model = None
        
        # 初始化模型
        self._init_quality_models()
    
    def _init_quality_models(self):
        """初始化质量评估模型"""
        # 文本质量模型
        try:
            # 可以使用预训练的语言模型评估流畅度
            # 或使用fineweb-edu风格的教育价值分类器
            self.logger.info("Quality models initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize quality models: {e}")
    
    def clean(self, items: List[Any]) -> List[CleanedDataItem]:
        """执行质量过滤"""
        cleaned_items = []
        
        for item in items:
            try:
                # 确定数据类型
                data_type = self._detect_data_type(item)
                
                # 计算质量评分
                if data_type == 'text':
                    quality_score = self.score_text_quality(item)
                    min_quality = self.quality_config.min_text_quality
                elif data_type == 'image':
                    quality_score = self.score_image_quality(item)
                    min_quality = self.quality_config.min_image_quality
                elif data_type == 'audio':
                    quality_score = self.score_audio_quality(item)
                    min_quality = self.quality_config.min_audio_quality
                else:
                    quality_score = None
                    min_quality = 0.0
                
                # 过滤低质量数据
                if quality_score and quality_score.composite < min_quality:
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type=data_type,
                        content=None,
                        status=CleaningStatus.FILTERED,
                        quality_score=quality_score.composite,
                        metadata={'reason': 'low_quality', 'scores': quality_score.to_dict()}
                    ))
                else:
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type=data_type,
                        content=self._extract_content(item),
                        status=CleaningStatus.SUCCESS,
                        quality_score=quality_score.composite if quality_score else None,
                        metadata={'scores': quality_score.to_dict() if quality_score else {}}
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error scoring quality: {e}")
                cleaned_items.append(CleanedDataItem(
                    item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                    original_id=getattr(item, 'data_id', ''),
                    data_type='unknown',
                    content=None,
                    status=CleaningStatus.FAILED,
                    errors=[str(e)]
                ))
        
        return cleaned_items
    
    def validate(self, item: Any) -> bool:
        """验证数据项"""
        return True  # 质量过滤器接受所有数据
    
    def score_text_quality(self, text: Any) -> TextQualityScore:
        """
        文本质量评分。
        使用fasttext分类器评估教育价值，
        使用perplexity评估流畅度。
        
        Args:
            text: 文本内容
            
        Returns:
            TextQualityScore: 文本质量评分
        """
        text_content = self._extract_text(text)
        
        if not text_content:
            return TextQualityScore(0.0, 0.0, 0.0, 0.0)
        
        # 流畅度评分（基于困惑度的反向）
        fluency = self._score_fluency(text_content)
        
        # 信息密度评分
        information_density = self._score_information_density(text_content)
        
        # 教育价值评分
        education_value = self._score_education_value(text_content)
        
        # 综合评分
        composite = (
            fluency * self.quality_config.text_fluency_weight +
            information_density * self.quality_config.text_information_density_weight +
            education_value * self.quality_config.text_education_value_weight
        )
        
        return TextQualityScore(
            fluency=fluency,
            information_density=information_density,
            education_value=education_value,
            composite=composite
        )
    
    def score_image_quality(self, image: Any) -> ImageQualityScore:
        """
        图像质量评分。
        使用CLIP aesthetic predictor、LAION-Aesthetics、NIMA模型。
        
        Args:
            image: 图像数据
            
        Returns:
            ImageQualityScore: 图像质量评分
        """
        # 简化实现，实际需要加载模型
        aesthetic = 0.5
        clarity = 0.5
        nsfw_score = 0.0
        
        # 综合评分
        composite = (
            aesthetic * self.quality_config.image_aesthetic_weight +
            clarity * self.quality_config.image_clarity_weight +
            (1.0 - nsfw_score) * self.quality_config.image_nsfw_weight
        )
        
        return ImageQualityScore(
            aesthetic=aesthetic,
            clarity=clarity,
            nsfw_score=nsfw_score,
            composite=composite
        )
    
    def score_audio_quality(self, audio: Any) -> AudioQualityScore:
        """
        音频质量评分。
        SNR + PESQ + 语音活动检测(VAD)。
        
        Args:
            audio: 音频数据
            
        Returns:
            AudioQualityScore: 音频质量评分
        """
        # 简化实现
        snr = 0.5
        clarity = 0.5
        speaker_quality = 0.5
        
        # 综合评分
        composite = (
            snr * self.quality_config.audio_snr_weight +
            clarity * self.quality_config.audio_clarity_weight +
            speaker_quality * self.quality_config.audio_speaker_quality_weight
        )
        
        return AudioQualityScore(
            snr=snr,
            clarity=clarity,
            speaker_quality=speaker_quality,
            composite=composite
        )
    
    def score_alignment(
        self,
        modality_a: Any,
        modality_b: Any,
        modality_type: str = 'image_text'
    ) -> float:
        """
        跨模态对齐质量评分。
        使用CLIP/CLAP模型。
        
        Args:
            modality_a: 模态A数据
            modality_b: 模态B数据
            modality_type: 模态类型
            
        Returns:
            float: 对齐评分 (0-1)
        """
        # 简化实现，实际需要CLIP/CLAP模型
        return 0.5
    
    def _score_fluency(self, text: str) -> float:
        """评估文本流畅度"""
        # 简化启发式实现
        # 实际应使用语言模型计算困惑度
        
        score = 0.5
        
        # 检查句子结构
        sentences = text.split('.')
        if len(sentences) > 1:
            score += 0.2
        
        # 检查标点符号使用
        if ',' in text or '，' in text:
            score += 0.1
        
        # 检查大写（英文）
        if any(c.isupper() for c in text):
            score += 0.1
        
        # 惩罚重复
        words = text.split()
        if len(words) > 0:
            unique_words = set(words)
            unique_ratio = len(unique_words) / len(words)
            score = score * (0.5 + 0.5 * unique_ratio)
        
        return min(1.0, max(0.0, score))
    
    def _score_information_density(self, text: str) -> float:
        """评估信息密度"""
        # 基于词汇多样性和长度
        words = text.split()
        
        if len(words) == 0:
            return 0.0
        
        # 词汇多样性
        unique_words = set(word.lower() for word in words)
        lexical_diversity = len(unique_words) / len(words)
        
        # 平均词长
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 归一化
        score = (lexical_diversity * 0.6 + min(avg_word_length / 10.0, 1.0) * 0.4)
        
        return min(1.0, max(0.0, score))
    
    def _score_education_value(self, text: str) -> float:
        """评估教育价值"""
        # 简化启发式评分
        # 实际应使用fineweb-edu风格分类器
        
        score = 0.5
        
        # 专业术语
        academic_patterns = [
            '研究', '分析', '理论', '方法', '结果',
            'research', 'analysis', 'theory', 'method', 'result'
        ]
        
        for pattern in academic_patterns:
            if pattern in text.lower():
                score += 0.05
        
        # 数据和数字
        import re
        numbers = re.findall(r'\d+', text)
        if len(numbers) > 3:
            score += 0.1
        
        # 引用标记
        if '参考文献' in text or 'reference' in text.lower():
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _detect_data_type(self, item: Any) -> str:
        """检测数据类型"""
        if isinstance(item, str):
            return 'text'
        elif isinstance(item, bytes):
            # 需要进一步检查
            return 'binary'
        elif isinstance(item, dict):
            if 'text' in item:
                return 'text'
            elif 'image' in item:
                return 'image'
            elif 'audio' in item:
                return 'audio'
        
        return 'unknown'
    
    def _extract_text(self, item: Any) -> Optional[str]:
        """提取文本内容"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('text') or item.get('content')
        else:
            return None
    
    def _extract_content(self, item: Any) -> Any:
        """提取内容"""
        if isinstance(item, dict):
            return item.get('content') or item.get('text') or item.get('image') or item.get('audio')
        else:
            return item