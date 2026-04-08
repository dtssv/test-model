"""
数据质量评分器
自动评估数据质量并打分
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
class QualityScorerConfig(LabelConfig):
    """质量评分器配置"""
    # 评分维度
    dimensions: List[str] = None  # fluency, relevance, coherence, informativeness, safety
    
    # 维度权重
    dimension_weights: Dict[str, float] = None
    
    # 评分范围
    min_score: float = 0.0
    max_score: float = 1.0
    pass_threshold: float = 0.6
    
    # 文本质量
    text_quality_model: str = "fairseq/wav2vec2-base"  # 或其他模型
    enable_perplexity: bool = True
    enable_fluency: bool = True
    enable_relevance: bool = True
    
    # 图像质量
    image_quality_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    enable_aesthetic_score: bool = True
    enable_clip_score: bool = True
    
    # 多模态对齐
    enable_alignment_score: bool = True
    alignment_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    # 模型配置
    use_gpu: bool = True
    batch_size: int = 32
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = ['fluency', 'relevance', 'coherence', 'informativeness', 'safety']
        
        if self.dimension_weights is None:
            # 默认权重：均匀分布
            num_dims = len(self.dimensions)
            self.dimension_weights = {dim: 1.0 / num_dims for dim in self.dimensions}


@dataclass
class QualityDimension:
    """质量维度评分"""
    name: str
    score: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'score': self.score,
            'details': self.details,
        }


class QualityScorer(BaseLabeler):
    """
    多维度数据质量评分器。
    支持文本、图像、多模态数据的质量评估。
    """
    
    def __init__(self, config: QualityScorerConfig):
        super().__init__(config)
        self.quality_config = config
        
        # 质量评估模型
        self.fluency_model = None
        self.perplexity_model = None
        self.clip_model = None
        self.aesthetic_model = None
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        """初始化质量评估模型"""
        try:
            # 困惑度模型（用于文本质量）
            if self.quality_config.enable_perplexity:
                self._init_perplexity_model()
            
            # CLIP模型（用于图像和多模态质量）
            if self.quality_config.enable_clip_score or self.quality_config.enable_alignment_score:
                self._init_clip_model()
            
            # 美学评分模型
            if self.quality_config.enable_aesthetic_score:
                self._init_aesthetic_model()
            
            self.logger.info("Quality scorer models initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quality models: {e}")
    
    def _init_perplexity_model(self):
        """初始化困惑度模型"""
        # 实际实现：
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
        # self.perplexity_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pass
    
    def _init_clip_model(self):
        """初始化CLIP模型"""
        # 实际实现：
        # import clip
        # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        pass
    
    def _init_aesthetic_model(self):
        """初始化美学评分模型"""
        # 实际实现：
        # from transformers import AutoModel, AutoProcessor
        # self.aesthetic_model = AutoModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        # self.aesthetic_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        pass
    
    def label(self, items: List[Any]) -> List[LabeledDataItem]:
        """
        为数据项评分。
        
        Args:
            items: 待评分的数据项
            
        Returns:
            List[LabeledDataItem]: 标注后的数据项
        """
        labeled_items = []
        
        for item in items:
            try:
                # 检测数据类型
                data_type = self._detect_data_type(item)
                
                # 根据数据类型选择评分方法
                if data_type == 'text':
                    dimension_scores = self._score_text(item)
                elif data_type == 'image':
                    dimension_scores = self._score_image(item)
                elif data_type == 'multimodal':
                    dimension_scores = self._score_multimodal(item)
                else:
                    dimension_scores = []
                
                # 计算综合评分
                overall_score = self._compute_overall_score(dimension_scores)
                
                # 创建标签
                label = Label(
                    label_type=LabelType.QUALITY_SCORE,
                    label_value={
                        'overall_score': overall_score,
                        'dimensions': [dim.to_dict() for dim in dimension_scores],
                        'passed': overall_score >= self.quality_config.pass_threshold,
                    },
                    confidence=overall_score,
                    metadata={
                        'data_type': data_type,
                        'scorer': 'QualityScorer',
                    }
                )
                
                status = LabelStatus.COMPLETED if overall_score >= self.quality_config.pass_threshold else LabelStatus.NEEDS_REVIEW
                
                labeled_items.append(self._create_labeled_item(
                    item,
                    [label],
                    status
                ))
                
            except Exception as e:
                self.logger.error(f"Error scoring item: {e}")
                labeled_items.append(self._create_labeled_item(
                    item,
                    [],
                    LabelStatus.FAILED,
                    [str(e)]
                ))
        
        return labeled_items
    
    def validate(self, item: Any) -> bool:
        """验证数据项"""
        return True  # 质量评分器接受所有数据
    
    def _score_text(self, item: Any) -> List[QualityDimension]:
        """
        文本质量评分。
        维度：流畅度、相关性、连贯性、信息量、安全性。
        """
        text = self._extract_text(item)
        
        if not text:
            return []
        
        dimensions = []
        
        # 流畅度评分
        if 'fluency' in self.quality_config.dimensions:
            fluency_score = self._score_fluency(text)
            dimensions.append(QualityDimension(
                name='fluency',
                score=fluency_score,
                details={'method': 'perplexity_based'}
            ))
        
        # 相关性评分
        if 'relevance' in self.quality_config.dimensions:
            relevance_score = self._score_relevance(text)
            dimensions.append(QualityDimension(
                name='relevance',
                score=relevance_score,
                details={'method': 'topic_modeling'}
            ))
        
        # 连贯性评分
        if 'coherence' in self.quality_config.dimensions:
            coherence_score = self._score_coherence(text)
            dimensions.append(QualityDimension(
                name='coherence',
                score=coherence_score,
                details={'method': 'sentence_similarity'}
            ))
        
        # 信息量评分
        if 'informativeness' in self.quality_config.dimensions:
            info_score = self._score_informativeness(text)
            dimensions.append(QualityDimension(
                name='informativeness',
                score=info_score,
                details={'method': 'lexical_diversity'}
            ))
        
        # 安全性评分
        if 'safety' in self.quality_config.dimensions:
            safety_score = self._score_safety(text)
            dimensions.append(QualityDimension(
                name='safety',
                score=safety_score,
                details={'method': 'toxicity_detection'}
            ))
        
        return dimensions
    
    def _score_image(self, item: Any) -> List[QualityDimension]:
        """
        图像质量评分。
        维度：美学评分、清晰度、CLIP评分。
        """
        dimensions = []
        
        # 美学评分
        if self.quality_config.enable_aesthetic_score and 'aesthetic' in self.quality_config.dimensions:
            aesthetic_score = self._score_aesthetic(item)
            dimensions.append(QualityDimension(
                name='aesthetic',
                score=aesthetic_score,
                details={'method': 'aesthetic_predictor'}
            ))
        
        # 清晰度评分
        if 'clarity' in self.quality_config.dimensions:
            clarity_score = self._score_clarity(item)
            dimensions.append(QualityDimension(
                name='clarity',
                score=clarity_score,
                details={'method': 'laplacian_variance'}
            ))
        
        return dimensions
    
    def _score_multimodal(self, item: Any) -> List[QualityDimension]:
        """
        多模态数据质量评分。
        包括文本评分、图像评分和对齐评分。
        """
        dimensions = []
        
        # 文本评分
        text = self._extract_text(item)
        if text:
            dimensions.extend(self._score_text(text))
        
        # 图像评分
        dimensions.extend(self._score_image(item))
        
        # 对齐评分
        if self.quality_config.enable_alignment_score and text:
            alignment_score = self._score_alignment(item)
            dimensions.append(QualityDimension(
                name='alignment',
                score=alignment_score,
                details={'method': 'clip_similarity'}
            ))
        
        return dimensions
    
    def _score_fluency(self, text: str) -> float:
        """评估文本流畅度"""
        # 简化实现：基于启发式规则
        score = 1.0
        
        # 句子完整性
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        # 平均句子长度
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        if avg_length < 5:
            score *= 0.7
        
        # 标点符号使用
        if not re.search(r'[。！？.!?]', text):
            score *= 0.8
        
        # 重复词检测
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            score *= (0.5 + 0.5 * unique_ratio)
        
        return max(self.quality_config.min_score, min(self.quality_config.max_score, score))
    
    def _score_relevance(self, text: str) -> float:
        """评估文本相关性"""
        # 简化实现：基于关键词密度
        score = 0.7
        
        # 检查是否有主题相关词
        topic_keywords = ['研究', '分析', '结论', '方法', '结果', 'research', 'analysis', 'method', 'result']
        keyword_count = sum(1 for keyword in topic_keywords if keyword in text.lower())
        
        score = min(1.0, 0.5 + keyword_count * 0.1)
        
        return score
    
    def _score_coherence(self, text: str) -> float:
        """评估文本连贯性"""
        # 简化实现：基于句子间连接词
        score = 0.7
        
        # 连接词检测
        connectors = ['因此', '所以', '但是', '然而', '此外', '首先', '其次', 
                      'therefore', 'so', 'but', 'however', 'moreover', 'first', 'second']
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        
        score = min(1.0, 0.6 + connector_count * 0.05)
        
        return score
    
    def _score_informativeness(self, text: str) -> float:
        """评估文本信息量"""
        # 基于词汇多样性和内容密度
        words = text.split()
        
        if len(words) == 0:
            return 0.0
        
        # 词汇多样性
        unique_words = set(word.lower() for word in words)
        lexical_diversity = len(unique_words) / len(words)
        
        # 信息密度（基于停用词比例）
        stopwords = {'的', '是', '在', '和', '了', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}
        content_words = [w for w in words if w.lower() not in stopwords]
        content_ratio = len(content_words) / len(words) if words else 0
        
        # 综合评分
        score = 0.4 * lexical_diversity + 0.6 * content_ratio
        
        return max(self.quality_config.min_score, min(self.quality_config.max_score, score))
    
    def _score_safety(self, text: str) -> float:
        """评估文本安全性"""
        # 简化实现：检测敏感词
        score = 1.0
        
        # 敏感词列表（示例）
        sensitive_words = ['暴力', '色情', '赌博', 'violence', 'porn', 'gambling']
        
        for word in sensitive_words:
            if word in text.lower():
                score *= 0.5
        
        return score
    
    def _score_aesthetic(self, item: Any) -> float:
        """评估图像美学评分"""
        # 简化实现，实际需要美学评分模型
        # 返回默认分数
        return 0.75
    
    def _score_clarity(self, item: Any) -> float:
        """评估图像清晰度"""
        # 简化实现，实际需要使用拉普拉斯方差等方法
        return 0.8
    
    def _score_alignment(self, item: Any) -> float:
        """评估多模态对齐度"""
        # 简化实现，实际需要使用CLIP模型计算相似度
        return 0.7
    
    def _compute_overall_score(self, dimensions: List[QualityDimension]) -> float:
        """计算综合评分"""
        if not dimensions:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for dim in dimensions:
            weight = self.quality_config.dimension_weights.get(dim.name, 1.0)
            total_score += dim.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        overall = total_score / total_weight
        
        return max(self.quality_config.min_score, min(self.quality_config.max_score, overall))
    
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
            has_text = 'text' in item or 'content' in item
            has_image = 'image' in item or 'image_path' in item or 'image_url' in item
            
            if has_text and has_image:
                return 'multimodal'
            elif has_image:
                return 'image'
            elif has_text:
                return 'text'
        
        return 'unknown'