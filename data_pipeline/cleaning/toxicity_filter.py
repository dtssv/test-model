"""
有毒内容过滤器
检测并过滤有毒、仇恨、攻击性内容
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re

from .base_cleaner import BaseCleaner, CleaningConfig, CleanedDataItem, CleaningStatus

logger = logging.getLogger(__name__)


@dataclass
class ToxicityFilterConfig(CleaningConfig):
    """有毒内容过滤器配置"""
    enable_toxicity_detection: bool = True
    enable_hate_speech_detection: bool = True
    enable_profanity_detection: bool = True
    enable_threat_detection: bool = True
    
    # 阈值设置
    toxicity_threshold: float = 0.7  # 有毒内容阈值
    hate_speech_threshold: float = 0.7  # 仇恨言论阈值
    profanity_threshold: float = 0.8  # 脏话阈值
    threat_threshold: float = 0.7  # 威胁阈值
    
    # 处理策略
    filter_mode: str = 'remove'  # 'remove' 或 'redact'
    redact_char: str = '*'
    
    # 模型配置
    use_perspective_api: bool = False
    perspective_api_key: Optional[str] = None
    use_local_model: bool = True
    model_name: str = 'unitary/toxic-bert'


@dataclass
class ToxicityScore:
    """有毒内容评分"""
    toxicity: float  # 总体毒性
    severe_toxicity: float  # 严重毒性
    identity_attack: float  # 身份攻击
    insult: float  # 侮辱
    profanity: float  # 脏话
    threat: float  # 威胁
    sexually_explicit: float  # 性暗示
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'toxicity': self.toxicity,
            'severe_toxicity': self.severe_toxicity,
            'identity_attack': self.identity_attack,
            'insult': self.insult,
            'profanity': self.profanity,
            'threat': self.threat,
            'sexually_explicit': self.sexually_explicit
        }
    
    def is_toxic(self, config: ToxicityFilterConfig) -> bool:
        """判断是否为有毒内容"""
        return (
            self.toxicity > config.toxicity_threshold or
            self.identity_attack > config.hate_speech_threshold or
            self.profanity > config.profanity_threshold or
            self.threat > config.threat_threshold
        )


class ToxicityFilter(BaseCleaner):
    """
    有毒内容检测与过滤器。
    支持多种检测方式：本地模型、Perspective API、规则匹配。
    """
    
    def __init__(self, config: ToxicityFilterConfig):
        super().__init__(config)
        self.toxicity_config = config
        
        # 有毒内容检测模型
        self.toxicity_model = None
        self.tokenizer = None
        
        # 中文敏感词列表
        self.chinese_profanity_words = set()
        # 英文敏感词列表
        self.english_profanity_words = set()
        
        # 初始化模型
        self._init_toxicity_detector()
        self._init_profanity_lists()
    
    def _init_toxicity_detector(self):
        """初始化有毒内容检测器"""
        if not self.toxicity_config.use_local_model:
            return
        
        try:
            # 实际实现需要加载transformers模型
            # self.tokenizer = AutoTokenizer.from_pretrained(self.toxicity_config.model_name)
            # self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            #     self.toxicity_config.model_name
            # )
            self.logger.info("Toxicity detector initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize toxicity detector: {e}")
    
    def _init_profanity_lists(self):
        """初始化敏感词列表"""
        # 常见中文敏感词（示例，实际应使用完整词库）
        self.chinese_profanity_words = {
            # 这里应该包含实际的敏感词列表
            # 为了演示，仅保留结构
        }
        
        # 常见英文敏感词
        self.english_profanity_words = {
            # 实际应包含完整词库
        }
    
    def clean(self, items: List[Any]) -> List[CleanedDataItem]:
        """执行有毒内容过滤"""
        cleaned_items = []
        
        for item in items:
            try:
                # 提取文本
                text_content = self._extract_text(item)
                
                if not text_content:
                    # 无文本内容，直接通过
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='unknown',
                        content=self._extract_content(item),
                        status=CleaningStatus.SUCCESS,
                        metadata={'toxicity_check': 'skipped'}
                    ))
                    continue
                
                # 检测有毒内容
                toxicity_score = self.detect_toxicity(text_content)
                
                # 判断是否过滤
                if toxicity_score.is_toxic(self.toxicity_config):
                    if self.toxicity_config.filter_mode == 'remove':
                        # 移除模式：直接过滤
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='text',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            metadata={
                                'reason': 'toxic_content',
                                'toxicity_score': toxicity_score.to_dict()
                            }
                        ))
                    else:
                        # 编辑模式：替换敏感内容
                        redacted_text = self.redact_toxic_content(
                            text_content,
                            toxicity_score
                        )
                        
                        if isinstance(item, dict):
                            item['text'] = redacted_text
                            item['content'] = redacted_text
                        else:
                            item = redacted_text
                        
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='text',
                            content=self._extract_content(item),
                            status=CleaningStatus.SUCCESS,
                            metadata={
                                'toxicity_score': toxicity_score.to_dict(),
                                'action': 'redacted'
                            }
                        ))
                else:
                    # 通过检测
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(id(item)))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='text',
                        content=self._extract_content(item),
                        status=CleaningStatus.SUCCESS,
                        metadata={'toxicity_score': toxicity_score.to_dict()}
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error in toxicity filtering: {e}")
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
        return True  # 有毒内容过滤器接受所有数据进行检测
    
    def detect_toxicity(self, text: str) -> ToxicityScore:
        """
        检测文本毒性。
        使用本地模型或Perspective API。
        
        Args:
            text: 文本内容
            
        Returns:
            ToxicityScore: 毒性评分
        """
        if self.toxicity_config.use_perspective_api:
            return self._detect_with_perspective_api(text)
        else:
            return self._detect_with_local_model(text)
    
    def _detect_with_local_model(self, text: str) -> ToxicityScore:
        """
        使用本地模型检测毒性。
        使用transformers的toxic-bert模型。
        
        Args:
            text: 文本内容
            
        Returns:
            ToxicityScore: 毒性评分
        """
        # 简化实现，返回默认评分
        # 实际实现应加载并运行模型
        
        toxicity = 0.1
        severe_toxicity = 0.05
        identity_attack = 0.05
        insult = 0.1
        profanity = 0.1
        threat = 0.05
        sexually_explicit = 0.05
        
        # 规则检测
        if self._contains_profanity(text):
            profanity = 0.9
        
        if self._contains_hate_speech(text):
            identity_attack = 0.8
            toxicity = max(toxicity, 0.8)
        
        if self._contains_threat(text):
            threat = 0.9
            toxicity = max(toxicity, 0.9)
        
        return ToxicityScore(
            toxicity=toxicity,
            severe_toxicity=severe_toxicity,
            identity_attack=identity_attack,
            insult=insult,
            profanity=profanity,
            threat=threat,
            sexually_explicit=sexually_explicit
        )
    
    def _detect_with_perspective_api(self, text: str) -> ToxicityScore:
        """
        使用Perspective API检测毒性。
        
        Args:
            text: 文本内容
            
        Returns:
            ToxicityScore: 毒性评分
        """
        # 简化实现
        # 实际应调用Google Perspective API
        
        return ToxicityScore(
            toxicity=0.1,
            severe_toxicity=0.05,
            identity_attack=0.05,
            insult=0.05,
            profanity=0.05,
            threat=0.05,
            sexually_explicit=0.05
        )
    
    def _contains_profanity(self, text: str) -> bool:
        """检查是否包含脏话"""
        text_lower = text.lower()
        
        # 检查中文敏感词
        for word in self.chinese_profanity_words:
            if word in text:
                return True
        
        # 检查英文敏感词
        for word in self.english_profanity_words:
            if word in text_lower:
                return True
        
        return False
    
    def _contains_hate_speech(self, text: str) -> bool:
        """检查是否包含仇恨言论"""
        # 仇恨言论模式
        hate_patterns = [
            r'讨厌.*种族',
            r'仇恨.*民族',
            r'hate\s+.*people',
            r'kill\s+all',
        ]
        
        for pattern in hate_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_threat(self, text: str) -> bool:
        """检查是否包含威胁"""
        # 威胁模式
        threat_patterns = [
            r'我要杀',
            r'你死定了',
            r'i\s+will\s+kill',
            r'you\s+will\s+die',
            r'威胁',
            r'threat',
        ]
        
        for pattern in threat_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def redact_toxic_content(
        self,
        text: str,
        toxicity_score: ToxicityScore
    ) -> str:
        """
        编辑有毒内容（替换敏感词）。
        
        Args:
            text: 原始文本
            toxicity_score: 毒性评分
            
        Returns:
            str: 编辑后的文本
        """
        redacted_text = text
        
        # 替换中文敏感词
        for word in self.chinese_profanity_words:
            if word in redacted_text:
                replacement = self.toxicity_config.redact_char * len(word)
                redacted_text = redacted_text.replace(word, replacement)
        
        # 替换英文敏感词
        for word in self.english_profanity_words:
            if word.lower() in redacted_text.lower():
                replacement = self.toxicity_config.redact_char * len(word)
                redacted_text = re.sub(
                    re.escape(word),
                    replacement,
                    redacted_text,
                    flags=re.IGNORECASE
                )
        
        return redacted_text
    
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
            return item.get('content') or item.get('text')
        else:
            return item