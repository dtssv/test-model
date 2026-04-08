"""
安全标签标注器
检测和标注数据安全性标签
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re

from .base_labeler import (
    BaseLabeler,
    LabelConfig,
    Label,
    LabelType,
    LabeledDataItem,
    LabelStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class SafetyLabelerConfig(LabelConfig):
    """安全标注器配置"""
    # 检测类型
    enable_nsfw_detection: bool = True
    enable_violence_detection: bool = True
    enable_hate_speech_detection: bool = True
    enable_self_harm_detection: bool = True
    enable_sexual_content_detection: bool = True
    
    # 阈值设置
    nsfw_threshold: float = 0.7
    violence_threshold: float = 0.7
    hate_speech_threshold: float = 0.7
    self_harm_threshold: float = 0.7
    sexual_content_threshold: float = 0.7
    
    # 模型配置
    text_model: str = "unitary/toxic-bert"
    image_model: str = "Falconsai/nsfw_image_detection"
    
    # 标签类型
    safety_categories: List[str] = None
    
    def __post_init__(self):
        if self.safety_categories is None:
            self.safety_categories = [
                'safe',
                'nsfw',
                'violence',
                'hate_speech',
                'self_harm',
                'sexual_content',
                'unsafe',
            ]


@dataclass
class SafetyAssessment:
    """安全评估结果"""
    is_safe: bool
    primary_risk: Optional[str]
    risk_scores: Dict[str, float]
    confidence: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_safe': self.is_safe,
            'primary_risk': self.primary_risk,
            'risk_scores': self.risk_scores,
            'confidence': self.confidence,
            'details': self.details,
        }


class SafetyLabeler(BaseLabeler):
    """
    安全标签标注器。
    检测文本和图像中的不安全内容。
    """
    
    def __init__(self, config: SafetyLabelerConfig):
        super().__init__(config)
        self.safety_config = config
        
        # 检测模型
        self.text_safety_model = None
        self.image_safety_model = None
        
        # 关键词列表
        self.offensive_keywords = self._init_offensive_keywords()
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        """初始化安全检测模型"""
        try:
            # 文本安全模型
            # from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # self.text_safety_model = AutoModelForSequenceClassification.from_pretrained(
            #     self.safety_config.text_model
            # )
            # self.tokenizer = AutoTokenizer.from_pretrained(self.safety_config.text_model)
            
            # 图像安全模型
            # from transformers import AutoImageProcessor, AutoModelForImageClassification
            # self.image_safety_model = AutoModelForImageClassification.from_pretrained(
            #     self.safety_config.image_model
            # )
            
            self.logger.info("Safety detection models initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize safety models: {e}")
    
    def _init_offensive_keywords(self) -> Dict[str, List[str]]:
        """初始化冒犯性关键词列表"""
        return {
            'violence': [
                '杀', '死', '暴力的', '血腥',
                'kill', 'death', 'violent', 'bloody',
            ],
            'hate_speech': [
                '种族歧视', '性别歧视', '仇恨',
                'racist', 'sexist', 'hate',
            ],
            'sexual_content': [
                '色情', '性',
                'porn', 'sex', 'nude',
            ],
            'self_harm': [
                '自杀', '自残',
                'suicide', 'self-harm',
            ],
        }
    
    def label(self, items: List[Any]) -> List[LabeledDataItem]:
        """
        为数据项添加安全标签。
        
        Args:
            items: 待标注的数据项
            
        Returns:
            List[LabeledDataItem]: 标注后的数据项
        """
        labeled_items = []
        
        for item in items:
            try:
                # 检测数据类型
                data_type = self._detect_data_type(item)
                
                # 根据数据类型进行安全评估
                if data_type == 'text':
                    assessment = self._assess_text_safety(item)
                elif data_type == 'image':
                    assessment = self._assess_image_safety(item)
                elif data_type == 'multimodal':
                    assessment = self._assess_multimodal_safety(item)
                else:
                    assessment = SafetyAssessment(
                        is_safe=True,
                        primary_risk=None,
                        risk_scores={},
                        confidence=1.0,
                        details={'message': 'Unknown data type, marked as safe'}
                    )
                
                # 创建标签
                label = Label(
                    label_type=LabelType.SAFETY_LABEL,
                    label_value=assessment.to_dict(),
                    confidence=assessment.confidence,
                    metadata={
                        'data_type': data_type,
                        'is_safe': assessment.is_safe,
                        'primary_risk': assessment.primary_risk,
                    }
                )
                
                # 确定状态
                status = LabelStatus.COMPLETED if assessment.is_safe else LabelStatus.NEEDS_REVIEW
                
                labeled_items.append(self._create_labeled_item(
                    item,
                    [label],
                    status
                ))
                
            except Exception as e:
                self.logger.error(f"Error assessing safety: {e}")
                labeled_items.append(self._create_labeled_item(
                    item,
                    [],
                    LabelStatus.FAILED,
                    [str(e)]
                ))
        
        return labeled_items
    
    def validate(self, item: Any) -> bool:
        """验证数据项"""
        return True
    
    def _assess_text_safety(self, item: Any) -> SafetyAssessment:
        """评估文本安全性"""
        text = self._extract_text(item)
        
        if not text:
            return SafetyAssessment(
                is_safe=True,
                primary_risk=None,
                risk_scores={},
                confidence=1.0,
                details={'message': 'No text content'}
            )
        
        risk_scores = {}
        
        # NSFW检测
        if self.safety_config.enable_nsfw_detection:
            risk_scores['nsfw'] = self._detect_nsfw_text(text)
        
        # 暴力内容检测
        if self.safety_config.enable_violence_detection:
            risk_scores['violence'] = self._detect_violence_text(text)
        
        # 仇恨言论检测
        if self.safety_config.enable_hate_speech_detection:
            risk_scores['hate_speech'] = self._detect_hate_speech_text(text)
        
        # 自残内容检测
        if self.safety_config.enable_self_harm_detection:
            risk_scores['self_harm'] = self._detect_self_harm_text(text)
        
        # 性内容检测
        if self.safety_config.enable_sexual_content_detection:
            risk_scores['sexual_content'] = self._detect_sexual_content_text(text)
        
        # 确定主要风险
        primary_risk = None
        max_risk_score = 0.0
        
        for risk_type, score in risk_scores.items():
            if score > max_risk_score:
                max_risk_score = score
                primary_risk = risk_type
        
        # 判断是否安全
        is_safe = all(
            score < getattr(self.safety_config, f'{risk}_threshold', 0.7)
            for risk, score in risk_scores.items()
        )
        
        # 计算置信度
        confidence = 1.0 - max_risk_score if is_safe else max_risk_score
        
        return SafetyAssessment(
            is_safe=is_safe,
            primary_risk=primary_risk,
            risk_scores=risk_scores,
            confidence=confidence,
            details={'method': 'keyword_and_model'}
        )
    
    def _assess_image_safety(self, item: Any) -> SafetyAssessment:
        """评估图像安全性"""
        risk_scores = {}
        
        # 简化实现：返回默认值
        # 实际应使用图像分类模型
        if self.safety_config.enable_nsfw_detection:
            risk_scores['nsfw'] = 0.1
        
        if self.safety_config.enable_violence_detection:
            risk_scores['violence'] = 0.1
        
        if self.safety_config.enable_sexual_content_detection:
            risk_scores['sexual_content'] = 0.1
        
        is_safe = all(score < 0.7 for score in risk_scores.values())
        
        return SafetyAssessment(
            is_safe=is_safe,
            primary_risk=None,
            risk_scores=risk_scores,
            confidence=0.9,
            details={'method': 'image_model'}
        )
    
    def _assess_multimodal_safety(self, item: Any) -> SafetyAssessment:
        """评估多模态数据安全性"""
        # 文本评估
        text_assessment = self._assess_text_safety(item)
        
        # 图像评估
        image_assessment = self._assess_image_safety(item)
        
        # 合并风险评分
        combined_risk_scores = {}
        
        for risk_type in set(text_assessment.risk_scores.keys()) | set(image_assessment.risk_scores.keys()):
            text_score = text_assessment.risk_scores.get(risk_type, 0.0)
            image_score = image_assessment.risk_scores.get(risk_type, 0.0)
            combined_risk_scores[risk_type] = max(text_score, image_score)
        
        # 确定是否安全
        is_safe = text_assessment.is_safe and image_assessment.is_safe
        
        # 确定主要风险
        primary_risk = text_assessment.primary_risk or image_assessment.primary_risk
        
        return SafetyAssessment(
            is_safe=is_safe,
            primary_risk=primary_risk,
            risk_scores=combined_risk_scores,
            confidence=min(text_assessment.confidence, image_assessment.confidence),
            details={
                'method': 'multimodal',
                'text_assessment': text_assessment.to_dict(),
                'image_assessment': image_assessment.to_dict(),
            }
        )
    
    def _detect_nsfw_text(self, text: str) -> float:
        """检测文本NSFW内容"""
        return self._keyword_detection(text, 'sexual_content')
    
    def _detect_violence_text(self, text: str) -> float:
        """检测文本暴力内容"""
        return self._keyword_detection(text, 'violence')
    
    def _detect_hate_speech_text(self, text: str) -> float:
        """检测文本仇恨言论"""
        return self._keyword_detection(text, 'hate_speech')
    
    def _detect_self_harm_text(self, text: str) -> float:
        """检测文本自残内容"""
        return self._keyword_detection(text, 'self_harm')
    
    def _detect_sexual_content_text(self, text: str) -> float:
        """检测文本性内容"""
        return self._keyword_detection(text, 'sexual_content')
    
    def _keyword_detection(self, text: str, category: str) -> float:
        """基于关键词的检测"""
        text_lower = text.lower()
        keywords = self.offensive_keywords.get(category, [])
        
        # 计算匹配数量
        match_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        # 归一化评分
        if match_count == 0:
            return 0.0
        elif match_count <= 2:
            return 0.3
        elif match_count <= 5:
            return 0.6
        else:
            return 0.9
    
    def _extract_text(self, item: Any) -> Optional[str]:
        """提取文本"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('text') or item.get('content') or item.get('caption')
        elif hasattr(item, 'text'):
            return item.text
        else:
            return None
    
    def _detect_data_type(self, item: Any) -> str:
        """检测数据类型"""
        if isinstance(item, str):
            return 'text'
        elif isinstance(item, dict):
            has_text = 'text' in item or 'content' in item or 'caption' in item
            has_image = 'image' in item or 'image_path' in item or 'image_url' in item
            
            if has_text and has_image:
                return 'multimodal'
            elif has_image:
                return 'image'
            elif has_text:
                return 'text'
        
        return 'unknown'