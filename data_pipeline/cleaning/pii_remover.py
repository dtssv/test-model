"""
个人隐私信息(PII)检测与移除
使用Microsoft Presidio框架 + 自定义中文规则
检测类型：姓名、邮箱、电话、身份证号、银行卡号、地址、IP地址
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    """PII实体"""
    entity_type: str  # 实体类型：PERSON, EMAIL, PHONE, ID_CARD, etc.
    text: str  # 原始文本
    start: int  # 起始位置
    end: int  # 结束位置
    score: float  # 置信度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_type': self.entity_type,
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'score': self.score
        }


@dataclass
class PIIConfig:
    """PII检测配置"""
    enable_presidio: bool = True
    enable_chinese_rules: bool = True
    anonymize_strategy: str = 'replace'  # replace, mask, hash
    replace_with: str = '[REDACTED]'  # 替换文本
    mask_char: str = '*'  # 遮盖字符
    mask_keep_first: int = 2  # 保留前N个字符
    mask_keep_last: int = 2  # 保留后N个字符
    min_confidence: float = 0.7  # 最小置信度阈值
    # 检测的实体类型
    entity_types: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = [
                'PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER',
                'ID_CARD', 'BANK_CARD', 'IP_ADDRESS',
                'LOCATION', 'ORGANIZATION'
            ]


class PIIRemover:
    """
    个人可识别信息(PII)检测与移除。
    使用Microsoft Presidio框架 + 自定义中文规则。
    检测类型：姓名、邮箱、电话、身份证号、银行卡号、地址、IP地址。
    """
    
    def __init__(self, config: PIIConfig):
        """
        初始化PII移除器
        
        Args:
            config: PII配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Presidio分析器和匿名化器
        self.analyzer = None
        self.anonymizer = None
        
        # 中文正则规则
        self.chinese_patterns = {}
        
        # 初始化
        if config.enable_presidio:
            self._init_presidio()
        
        if config.enable_chinese_rules:
            self._init_chinese_rules()
    
    def _init_presidio(self):
        """初始化Presidio分析器"""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.logger.info("Presidio analyzer and anonymizer initialized")
            
        except ImportError:
            self.logger.warning("Presidio not installed. Install with: pip install presidio-analyzer presidio-anonymizer")
    
    def _init_chinese_rules(self):
        """初始化中文PII检测规则"""
        # 中国手机号
        self.chinese_patterns['PHONE_NUMBER'] = re.compile(
            r'(?:\+?86)?1[3-9]\d{9}'
        )
        
        # 中国身份证号
        self.chinese_patterns['ID_CARD'] = re.compile(
            r'\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b'
        )
        
        # 银行卡号（16-19位）
        self.chinese_patterns['BANK_CARD'] = re.compile(
            r'\b\d{16,19}\b'
        )
        
        # 中文姓名（简化规则）
        self.chinese_patterns['PERSON'] = re.compile(
            r'[\u4e00-\u9fa5]{2,4}(?:说|表示|认为|称|指出|透露|表示)'
        )
        
        # IP地址
        self.chinese_patterns['IP_ADDRESS'] = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        )
        
        # 邮箱
        self.chinese_patterns['EMAIL_ADDRESS'] = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        self.logger.info("Chinese PII patterns initialized")
    
    def detect_pii(self, text: str) -> List[PIIEntity]:
        """
        检测文本中的PII实体
        
        Args:
            text: 文本内容
            
        Returns:
            List[PIIEntity]: PII实体列表
        """
        entities = []
        
        # 使用Presidio检测
        if self.analyzer and self.config.enable_presidio:
            entities.extend(self._detect_with_presidio(text))
        
        # 使用中文规则检测
        if self.config.enable_chinese_rules:
            entities.extend(self._detect_with_chinese_rules(text))
        
        # 去重和合并
        entities = self._merge_entities(entities)
        
        return entities
    
    def _detect_with_presidio(self, text: str) -> List[PIIEntity]:
        """使用Presidio检测PII"""
        if not self.analyzer:
            return []
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language='en',  # Presidio主要支持英文
                entities=self.config.entity_types,
                minimum_score=self.config.min_confidence
            )
            
            entities = []
            for result in results:
                entities.append(PIIEntity(
                    entity_type=result.entity_type,
                    text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    score=result.score
                ))
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error detecting PII with Presidio: {e}")
            return []
    
    def _detect_with_chinese_rules(self, text: str) -> List[PIIEntity]:
        """使用中文规则检测PII"""
        entities = []
        
        for entity_type, pattern in self.chinese_patterns.items():
            for match in pattern.finditer(text):
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=0.9  # 规则匹配给较高置信度
                ))
        
        return entities
    
    def _merge_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """合并重叠的实体"""
        if not entities:
            return []
        
        # 按起始位置排序
        entities.sort(key=lambda e: e.start)
        
        merged = []
        for entity in entities:
            # 检查是否与已有实体重叠
            is_overlapping = False
            for existing in merged:
                if (entity.start >= existing.start and entity.start < existing.end) or \
                   (entity.end > existing.start and entity.end <= existing.end):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                merged.append(entity)
        
        return merged
    
    def anonymize(self, text: str, strategy: str = None) -> str:
        """
        匿名化处理
        
        Args:
            text: 原始文本
            strategy: 匿名化策略 ('replace', 'mask', 'hash')
            
        Returns:
            str: 匿名化后的文本
        """
        strategy = strategy or self.config.anonymize_strategy
        
        # 检测PII
        entities = self.detect_pii(text)
        
        if not entities:
            return text
        
        # 按位置倒序排序，从后往前替换
        entities.sort(key=lambda e: e.start, reverse=True)
        
        result = text
        for entity in entities:
            if strategy == 'replace':
                # 替换为占位符
                replacement = f'[{entity.entity_type}]'
                result = result[:entity.start] + replacement + result[entity.end:]
            
            elif strategy == 'mask':
                # 部分遮盖
                text_len = len(entity.text)
                keep_first = self.config.mask_keep_first
                keep_last = self.config.mask_keep_last
                
                if text_len <= keep_first + keep_last:
                    # 文本太短，全部遮盖
                    replacement = self.config.mask_char * text_len
                else:
                    # 遮盖中间部分
                    replacement = (
                        entity.text[:keep_first] +
                        self.config.mask_char * (text_len - keep_first - keep_last) +
                        entity.text[-keep_last:]
                    )
                
                result = result[:entity.start] + replacement + result[entity.end:]
            
            elif strategy == 'hash':
                # 哈希替换
                import hashlib
                hashed = hashlib.md5(entity.text.encode()).hexdigest()[:8]
                replacement = f'[{entity.entity_type}_{hashed}]'
                result = result[:entity.start] + replacement + result[entity.end:]
            
            else:
                # 默认替换
                replacement = self.config.replace_with
                result = result[:entity.start] + replacement + result[entity.end:]
        
        return result
    
    def detect_chinese_pii(self, text: str) -> List[PIIEntity]:
        """
        使用正则+NER模型检测中文PII
        
        Args:
            text: 文本内容
            
        Returns:
            List[PIIEntity]: PII实体列表
        """
        entities = []
        
        # 手机号检测
        for match in self.chinese_patterns['PHONE_NUMBER'].finditer(text):
            entities.append(PIIEntity(
                entity_type='PHONE_NUMBER',
                text=match.group(),
                start=match.start(),
                end=match.end(),
                score=0.95
            ))
        
        # 身份证号检测
        for match in self.chinese_patterns['ID_CARD'].finditer(text):
            entities.append(PIIEntity(
                entity_type='ID_CARD',
                text=match.group(),
                start=match.start(),
                end=match.end(),
                score=0.95
            ))
        
        # 银行卡号检测（需要进一步验证）
        for match in self.chinese_patterns['BANK_CARD'].finditer(text):
            card_number = match.group()
            # 使用Luhn算法验证银行卡号
            if self._validate_bank_card(card_number):
                entities.append(PIIEntity(
                    entity_type='BANK_CARD',
                    text=card_number,
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        
        # IP地址检测
        for match in self.chinese_patterns['IP_ADDRESS'].finditer(text):
            ip = match.group()
            if self._validate_ip_address(ip):
                entities.append(PIIEntity(
                    entity_type='IP_ADDRESS',
                    text=ip,
                    start=match.start(),
                    end=match.end(),
                    score=0.9
                ))
        
        return entities
    
    def _validate_bank_card(self, card_number: str) -> bool:
        """使用Luhn算法验证银行卡号"""
        digits = [int(d) for d in card_number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        
        checksum = sum(odd_digits)
        for d in even_digits:
            d *= 2
            if d > 9:
                d -= 9
            checksum += d
        
        return checksum % 10 == 0
    
    def _validate_ip_address(self, ip: str) -> bool:
        """验证IP地址"""
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    return False
            except ValueError:
                return False
        
        return True
    
    def redact_text(self, text: str) -> Tuple[str, List[PIIEntity]]:
        """
        检测并移除PII（便捷方法）
        
        Args:
            text: 原始文本
            
        Returns:
            Tuple[str, List[PIIEntity]]: (清理后的文本, PII实体列表)
        """
        entities = self.detect_pii(text)
        cleaned_text = self.anonymize(text)
        return cleaned_text, entities