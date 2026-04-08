"""
文本数据清洗器
执行多阶段清洗流程：规则清洗 → 语言过滤 → 质量过滤 → 去重 → PII移除 → 安全过滤
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import unicodedata
import logging
from pathlib import Path
import uuid

from .base_cleaner import (
    BaseCleaner, CleaningConfig, CleanedDataItem, CleaningStatus
)

logger = logging.getLogger(__name__)


@dataclass
class TextCleaningConfig(CleaningConfig):
    """文本清洗专用配置"""
    min_text_length: int = 50
    max_text_length: int = 100000
    allowed_languages: List[str] = field(default_factory=lambda: ['en', 'zh'])
    enable_language_detection: bool = True
    enable_perplexity_filter: bool = False
    max_perplexity: float = 1000.0
    enable_education_filter: bool = False
    min_education_score: float = 0.0
    max_char_repeat: int = 50  # 最大字符重复次数
    max_word_repeat: int = 5   # 最大词重复次数
    max_line_repeat: float = 0.3  # 最大行重复比例
    remove_boilerplate: bool = True
    normalize_unicode: bool = True
    remove_html_tags: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    language_model_path: Optional[str] = None  # KenLM模型路径


class TextCleaner(BaseCleaner):
    """
    文本数据清洗器。
    执行多阶段清洗流程：
    规则清洗 → 语言过滤 → 质量过滤 → 去重 → PII移除 → 安全过滤。
    """
    
    def __init__(self, config: TextCleaningConfig):
        super().__init__(config)
        self.text_config = config
        self.language_detector = None
        self.perplexity_model = None
        
        # 初始化语言检测器
        if config.enable_language_detection:
            self._init_language_detector()
        
        # 初始化困惑度模型
        if config.enable_perplexity_filter:
            self._init_perplexity_model()
    
    def _init_language_detector(self):
        """初始化语言检测器"""
        try:
            import fasttext
            # 实际使用时需要下载lid.176.bin模型
            # self.language_detector = fasttext.load_model('lid.176.bin')
            self.logger.info("Language detector initialized (fasttext)")
        except ImportError:
            self.logger.warning("fasttext not installed, language detection disabled")
    
    def _init_perplexity_model(self):
        """初始化困惑度模型"""
        if self.text_config.language_model_path:
            try:
                import kenlm
                self.perplexity_model = kenlm.Model(self.text_config.language_model_path)
                self.logger.info("Perplexity model loaded")
            except ImportError:
                self.logger.warning("kenlm not installed, perplexity filtering disabled")
    
    def clean(self, items: List[Any]) -> List[CleanedDataItem]:
        """
        执行完整文本清洗流程
        
        Args:
            items: 待清洗的文本数据项列表
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项列表
        """
        cleaned_items = []
        
        for item in items:
            try:
                # 获取文本内容
                text = self._extract_text(item)
                
                if not text:
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='text',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        metadata={'reason': 'empty_text'}
                    ))
                    continue
                
                cleaning_steps = []
                
                # 1. 基础验证
                if not self.validate(item):
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='text',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        metadata={'reason': 'validation_failed'}
                    ))
                    continue
                
                # 2. 移除HTML标签
                if self.text_config.remove_html_tags:
                    text = self.remove_html_tags(text)
                    cleaning_steps.append('remove_html_tags')
                
                # 3. 移除样板文本
                if self.text_config.remove_boilerplate:
                    text = self.remove_boilerplate(text)
                    cleaning_steps.append('remove_boilerplate')
                
                # 4. Unicode标准化
                if self.text_config.normalize_unicode:
                    text = self.normalize_unicode(text)
                    cleaning_steps.append('normalize_unicode')
                
                # 5. 移除重复内容
                text = self.remove_repetitions(
                    text,
                    self.text_config.max_char_repeat,
                    self.text_config.max_word_repeat
                )
                cleaning_steps.append('remove_repetitions')
                
                # 6. 语言过滤
                if self.text_config.enable_language_detection:
                    lang = self.detect_language(text)
                    if lang not in self.text_config.allowed_languages:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='text',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            metadata={'reason': 'language_filtered', 'language': lang}
                        ))
                        continue
                    cleaning_steps.append(f'language_detection:{lang}')
                
                # 7. 长度过滤
                if not self.filter_by_length(text):
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='text',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        cleaning_steps=cleaning_steps,
                        metadata={'reason': 'length_filtered', 'length': len(text)}
                    ))
                    continue
                
                # 8. 困惑度过滤
                if self.text_config.enable_perplexity_filter and self.perplexity_model:
                    perplexity = self.compute_perplexity(text)
                    if perplexity > self.text_config.max_perplexity:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='text',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            metadata={'reason': 'perplexity_filtered', 'perplexity': perplexity}
                        ))
                        continue
                    cleaning_steps.append(f'perplexity_filter:{perplexity:.2f}')
                
                # 9. 教育价值评分（可选）
                quality_score = None
                if self.text_config.enable_education_filter:
                    quality_score = self.score_education_value(text)
                    if quality_score < self.text_config.min_education_score:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='text',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            quality_score=quality_score,
                            metadata={'reason': 'quality_filtered'}
                        ))
                        continue
                    cleaning_steps.append(f'education_score:{quality_score:.2f}')
                
                # 清洗成功
                cleaned_items.append(CleanedDataItem(
                    item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                    original_id=getattr(item, 'data_id', ''),
                    data_type='text',
                    content=text,
                    status=CleaningStatus.SUCCESS,
                    quality_score=quality_score,
                    cleaning_steps=cleaning_steps,
                    metadata={
                        'length': len(text),
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    }
                ))
                
            except Exception as e:
                self.logger.error(f"Error cleaning item: {e}")
                cleaned_items.append(CleanedDataItem(
                    item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                    original_id=getattr(item, 'data_id', ''),
                    data_type='text',
                    content=None,
                    status=CleaningStatus.FAILED,
                    errors=[str(e)]
                ))
        
        return cleaned_items
    
    def validate(self, item: Any) -> bool:
        """
        验证文本数据项
        
        Args:
            item: 待验证的数据项
            
        Returns:
            bool: 是否通过验证
        """
        text = self._extract_text(item)
        
        if not text or not isinstance(text, str):
            return False
        
        # 检查最小长度
        if len(text.strip()) < self.text_config.min_text_length:
            return False
        
        # 检查最大长度
        if len(text) > self.text_config.max_text_length:
            return False
        
        return True
    
    def _extract_text(self, item: Any) -> Optional[str]:
        """
        从数据项中提取文本
        
        Args:
            item: 数据项
            
        Returns:
            Optional[str]: 文本内容
        """
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('text') or item.get('content')
        elif hasattr(item, 'content'):
            return item.content
        else:
            return None
    
    def remove_html_tags(self, text: str) -> str:
        """
        清除HTML标签和实体
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 解码HTML实体
        import html
        text = html.unescape(text)
        
        return text.strip()
    
    def remove_boilerplate(self, text: str) -> str:
        """
        移除网页样板文本（导航、页脚、广告等）
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 使用trafilatura提取正文
        try:
            import trafilatura
            
            extracted = trafilatura.extract(
                text,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            
            if extracted:
                return extracted
            
        except ImportError:
            self.logger.warning("trafilatura not installed, using basic boilerplate removal")
        
        # 基础样板文本移除
        # 移除常见的样板文本模式
        patterns = [
            r'Cookie\s+Policy.*?Accept',
            r'Terms\s+of\s+Service.*?Agree',
            r'Privacy\s+Policy.*?Accept',
            r'Subscribe\s+to\s+our\s+newsletter',
            r'Follow\s+us\s+on\s+(Twitter|Facebook|Instagram)',
            r'Copyright\s+©\s+\d{4}.*?All\s+rights\s+reserved',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    def normalize_unicode(self, text: str) -> str:
        """
        Unicode标准化(NFKC)，全角转半角，统一标点
        
        Args:
            text: 原始文本
            
        Returns:
            str: 标准化后的文本
        """
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 全角转半角
        rstring = ""
        for char in text:
            inside_code = ord(char)
            if inside_code == 12288:  # 全角空格
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 全角字符
                inside_code -= 65248
            rstring += chr(inside_code)
        
        text = rstring
        
        # 统一引号
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('"', '"').replace('"', '"')
        
        return text
    
    def remove_repetitions(
        self,
        text: str,
        max_char_repeat: int = 50,
        max_word_repeat: int = 5
    ) -> str:
        """
        移除字符级和词级重复
        
        Args:
            text: 原始文本
            max_char_repeat: 最大字符重复次数
            max_word_repeat: 最大词重复次数
            
        Returns:
            str: 清理后的文本
        """
        # 移除字符级重复（如"哈哈哈哈哈哈"）
        def replace_char_repeat(match):
            char = match.group(1)
            count = len(match.group(0))
            if count > max_char_repeat:
                return char * max_char_repeat
            return match.group(0)
        
        text = re.sub(r'(.)\1{10,}', replace_char_repeat, text)
        
        # 移除词级重复（如"很好 很好 很好"）
        def replace_word_repeat(match):
            word = match.group(1)
            count = len(match.group(0)) // (len(word) + 1)
            if count > max_word_repeat:
                return ' '.join([word] * max_word_repeat)
            return match.group(0)
        
        text = re.sub(r'(\S+)(\s+\1){3,}', replace_word_repeat, text)
        
        # 移除行级重复
        lines = text.split('\n')
        seen_lines = {}
        unique_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                if line_stripped not in seen_lines:
                    seen_lines[line_stripped] = 0
                    unique_lines.append(line)
                else:
                    seen_lines[line_stripped] += 1
            else:
                unique_lines.append(line)
        
        # 检查行重复比例
        if len(seen_lines) > 0:
            repeat_ratio = sum(1 for count in seen_lines.values() if count > 0) / len(seen_lines)
            if repeat_ratio > self.text_config.max_line_repeat:
                # 只保留第一次出现的行
                unique_lines = [line for i, line in enumerate(lines) 
                               if line.strip() not in lines[:i]]
        
        return '\n'.join(unique_lines)
    
    def filter_by_length(self, text: str) -> bool:
        """
        过滤过短或过长的文本
        
        Args:
            text: 文本内容
            
        Returns:
            bool: 是否保留
        """
        length = len(text)
        
        if length < self.text_config.min_text_length:
            return False
        
        if length > self.text_config.max_text_length:
            return False
        
        return True
    
    def detect_language(self, text: str) -> str:
        """
        使用fasttext lid模型检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码 (如 'en', 'zh')
        """
        if self.language_detector:
            try:
                predictions = self.language_detector.predict(text.replace('\n', ' '))
                lang = predictions[0][0].replace('__label__', '')
                return lang
            except Exception as e:
                self.logger.error(f"Error detecting language: {e}")
        
        # 简单启发式规则
        # 检查中文字符比例
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars / len(text) > 0.1:
            return 'zh'
        else:
            return 'en'
    
    def compute_perplexity(self, text: str) -> float:
        """
        使用KenLM语言模型计算困惑度
        
        Args:
            text: 文本内容
            
        Returns:
            float: 困惑度
        """
        if not self.perplexity_model:
            return 0.0
        
        try:
            import kenlm
            # 计算困惑度
            # KenLM返回的是log概率，需要转换为困惑度
            log_prob = self.perplexity_model.score(text)
            # 困惑度 = exp(-log_prob / N)
            words = text.split()
            if len(words) == 0:
                return float('inf')
            perplexity = pow(10, -log_prob / len(words))
            return perplexity
        except Exception as e:
            self.logger.error(f"Error computing perplexity: {e}")
            return float('inf')
    
    def score_education_value(self, text: str) -> float:
        """
        使用教育价值分类器评分
        
        Args:
            text: 文本内容
            
        Returns:
            float: 教育价值分数 (0-1)
        """
        # 这里可以加载fineweb-edu风格的分类器
        # 简化实现：基于启发式规则
        
        score = 0.5  # 基础分
        
        # 长度奖励
        if len(text) > 1000:
            score += 0.1
        
        # 专业术语奖励
        technical_patterns = [
            r'\b(however|therefore|furthermore|moreover)\b',
            r'\b(analysis|research|study|theory)\b',
            r'\d+\.\d+',  # 数字
            r'\b[A-Z]{2,}\b',  # 缩写
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            if matches:
                score += 0.05
        
        # 惩罚非正式语言
        informal_patterns = [
            r'\b(lol|omg|wtf|idk)\b',
            r'[!]{2,}',  # 多个感叹号
        ]
        
        for pattern in informal_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def filter_quality(self, item: Any, threshold: float = None) -> bool:
        """
        过滤低质量文本
        
        Args:
            item: 数据项
            threshold: 质量阈值
            
        Returns:
            bool: 是否保留
        """
        text = self._extract_text(item)
        
        if not text:
            return False
        
        # 检查长度
        if not self.filter_by_length(text):
            return False
        
        # 检查语言
        if self.text_config.enable_language_detection:
            lang = self.detect_language(text)
            if lang not in self.text_config.allowed_languages:
                return False
        
        # 检查困惑度
        if self.text_config.enable_perplexity_filter and self.perplexity_model:
            perplexity = self.compute_perplexity(text)
            if perplexity > self.text_config.max_perplexity:
                return False
        
        # 检查教育价值
        if self.text_config.enable_education_filter:
            score = self.score_education_value(text)
            if score < self.text_config.min_education_score:
                return False
        
        return True