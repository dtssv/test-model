"""
QA对生成器
自动从文本生成问答对
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import json
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
class QALabelerConfig(LabelConfig):
    """QA对生成器配置"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_questions: int = 3  # 每段文本生成多少个问题
    
    # 问题类型
    question_types: List[str] = None  # factual, inferential, conceptual, application
    difficulty_levels: List[str] = None  # easy, medium, hard
    
    # 质量控制
    min_question_length: int = 10
    max_question_length: int = 200
    min_answer_length: int = 20
    max_answer_length: int = 1000
    filter_duplicates: bool = True
    require_context_match: bool = True
    
    # 多样性
    enable_diverse_questions: bool = True
    diversity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.question_types is None:
            self.question_types = ['factual', 'inferential', 'conceptual']
        if self.difficulty_levels is None:
            self.difficulty_levels = ['easy', 'medium', 'hard']


class QALabeler(BaseLabeler):
    """
    QA对生成器。
    从文本自动生成问答对，支持多种问题类型和难度级别。
    """
    
    def __init__(self, config: QALabelerConfig):
        super().__init__(config)
        self.qa_config = config
        
        # 问题模板
        self.question_templates = self._init_question_templates()
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化语言模型"""
        try:
            # 实际实现需要加载transformers模型
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(self.qa_config.model_name)
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     self.qa_config.model_name,
            #     torch_dtype="auto",
            #     device_map="auto"
            # )
            self.logger.info(f"QA model initialized: {self.qa_config.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize QA model: {e}")
            self.model = None
    
    def _init_question_templates(self) -> Dict[str, List[str]]:
        """初始化问题模板"""
        return {
            'factual': [
                "What is {entity}?",
                "When did {event} happen?",
                "Who is {person}?",
                "Where is {location}?",
                "How many {quantity} are there?",
                "什么{是}{entity}？",
                "{event}发生在什么时候？",
                "{person}是谁？",
                "{location}在哪里？",
            ],
            'inferential': [
                "Why did {event} happen?",
                "What caused {result}?",
                "How does {process} work?",
                "What is the relationship between {A} and {B}?",
                "为什么{event}会发生？",
                "是什么导致了{result}？",
                "{process}是如何工作的？",
            ],
            'conceptual': [
                "What is the main idea of {topic}?",
                "Explain the concept of {concept}.",
                "Compare and contrast {A} and {B}.",
                "{topic}的主要思想是什么？",
                "解释{concept}的概念。",
                "比较和对比{A}和{B}。",
            ],
            'application': [
                "How can {knowledge} be applied to {situation}?",
                "What would happen if {condition}?",
                "How would you solve {problem} using {method}?",
                "如何将{knowledge}应用到{situation}？",
                "如果{condition}会发生什么？",
            ],
        }
    
    def label(self, items: List[Any]) -> List[LabeledDataItem]:
        """
        为文本生成QA对标注。
        
        Args:
            items: 包含文本的数据项
            
        Returns:
            List[LabeledDataItem]: 标注后的数据项
        """
        labeled_items = []
        
        for item in items:
            try:
                # 验证数据
                if not self.validate(item):
                    labeled_items.append(self._create_labeled_item(
                        item,
                        [],
                        LabelStatus.FAILED,
                        ['Invalid item for QA generation']
                    ))
                    continue
                
                # 提取文本
                text = self._extract_text(item)
                
                if not text or len(text) < self.qa_config.min_answer_length:
                    labeled_items.append(self._create_labeled_item(
                        item,
                        [],
                        LabelStatus.FAILED,
                        ['Text too short for QA generation']
                    ))
                    continue
                
                # 检查缓存
                cache_key = self._get_cache_key(item)
                if cache_key:
                    cached_label = self._get_cached_label(cache_key)
                    if cached_label:
                        labeled_items.append(self._create_labeled_item(
                            item,
                            [cached_label],
                            LabelStatus.COMPLETED
                        ))
                        continue
                
                # 生成QA对
                qa_pairs = self.generate_qa_pairs(text)
                
                # 过滤和质量检查
                if self.qa_config.filter_duplicates:
                    qa_pairs = self._filter_duplicates(qa_pairs)
                
                qa_pairs = self._quality_check(qa_pairs)
                
                # 创建标签
                labels = []
                for idx, qa in enumerate(qa_pairs):
                    label = Label(
                        label_type=LabelType.QA_PAIR,
                        label_value={
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'question_type': qa.get('question_type', 'unknown'),
                            'difficulty': qa.get('difficulty', 'medium'),
                            'context': qa.get('context', ''),
                        },
                        confidence=qa.get('confidence', 1.0),
                        metadata={
                            'model': self.qa_config.model_name,
                            'qa_index': idx,
                        }
                    )
                    labels.append(label)
                
                # 缓存
                if labels and cache_key:
                    self._cache_label(cache_key, labels[0])
                
                labeled_items.append(self._create_labeled_item(
                    item,
                    labels,
                    LabelStatus.COMPLETED if labels else LabelStatus.FAILED,
                    [] if labels else ['No valid QA pairs generated']
                ))
                
            except Exception as e:
                self.logger.error(f"Error generating QA pairs: {e}")
                labeled_items.append(self._create_labeled_item(
                    item,
                    [],
                    LabelStatus.FAILED,
                    [str(e)]
                ))
        
        return labeled_items
    
    def validate(self, item: Any) -> bool:
        """验证数据项是否包含有效文本"""
        if isinstance(item, str):
            return len(item) >= self.qa_config.min_answer_length
        elif isinstance(item, dict):
            text = item.get('text') or item.get('content')
            return text is not None and len(text) >= self.qa_config.min_answer_length
        elif hasattr(item, 'text'):
            return len(item.text) >= self.qa_config.min_answer_length
        
        return False
    
    def generate_qa_pairs(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本生成QA对。
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict]: QA对列表
        """
        qa_pairs = []
        
        try:
            # 方法1: 使用LLM生成
            if self.model is not None:
                qa_pairs = self._generate_with_llm(text)
            else:
                # 方法2: 使用规则和启发式方法
                qa_pairs = self._generate_with_heuristics(text)
            
            # 多样性过滤
            if self.qa_config.enable_diverse_questions:
                qa_pairs = self._ensure_diversity(qa_pairs)
            
        except Exception as e:
            self.logger.error(f"Error in QA generation: {e}")
        
        return qa_pairs
    
    def _generate_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """使用LLM生成QA对"""
        # 实际实现需要调用模型
        # prompt = self._build_generation_prompt(text)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_new_tokens=self.qa_config.max_new_tokens)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # qa_pairs = self._parse_llm_response(response)
        
        # 简化实现
        qa_pairs = []
        for i in range(self.qa_config.num_questions):
            question_type = self.qa_config.question_types[i % len(self.qa_config.question_types)]
            difficulty = self.qa_config.difficulty_levels[i % len(self.qa_config.difficulty_levels)]
            
            qa_pairs.append({
                'question': f"Generated question {i+1} about the text",
                'answer': "Generated answer based on the text content",
                'question_type': question_type,
                'difficulty': difficulty,
                'context': text[:200],
                'confidence': 0.85
            })
        
        return qa_pairs
    
    def _generate_with_heuristics(self, text: str) -> List[Dict[str, Any]]:
        """使用启发式方法生成QA对"""
        qa_pairs = []
        
        # 分割文本为句子
        sentences = self._split_sentences(text)
        
        # 从句子中提取关键信息
        for i, sentence in enumerate(sentences[:self.qa_config.num_questions]):
            # 提取实体
            entities = self._extract_entities(sentence)
            
            if entities:
                # 生成事实性问题
                entity = entities[0]
                question = f"What is {entity}?"
                answer = sentence
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'question_type': 'factual',
                    'difficulty': 'easy',
                    'context': sentence,
                    'confidence': 0.7
                })
            else:
                # 生成概念性问题
                question = f"What does this sentence mean: {sentence[:50]}...?"
                answer = sentence
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'question_type': 'conceptual',
                    'difficulty': 'medium',
                    'context': sentence,
                    'confidence': 0.6
                })
        
        return qa_pairs
    
    def _build_generation_prompt(self, text: str) -> str:
        """构建生成提示词"""
        prompt = f"""Based on the following text, generate {self.qa_config.num_questions} question-answer pairs.

Text:
{text}

Requirements:
1. Questions should cover different types: {', '.join(self.qa_config.question_types)}
2. Questions should have different difficulty levels: {', '.join(self.qa_config.difficulty_levels)}
3. Answers must be directly supported by the text
4. Each QA pair should include: question, answer, question_type, difficulty

Output format (JSON):
[
  {{
    "question": "...",
    "answer": "...",
    "question_type": "...",
    "difficulty": "..."
  }},
  ...
]
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                qa_pairs = json.loads(json_match.group())
                return qa_pairs
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
        
        return []
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割文本为句子"""
        # 中文和英文句子分割
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体（简化实现）"""
        # 实际应使用NER模型
        entities = []
        
        # 简单规则匹配
        # 人名（中文）
        person_pattern = r'[A-Z][a-z]+\s[A-Z][a-z]+|[A-Z][a-z]+'
        entities.extend(re.findall(person_pattern, text))
        
        # 数字
        number_pattern = r'\d+(?:\.\d+)?(?:万|亿|百万)?'
        numbers = re.findall(number_pattern, text)
        if numbers:
            entities.append(f"数字{numbers[0]}")
        
        return entities
    
    def _filter_duplicates(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤重复的QA对"""
        seen_questions = set()
        unique_pairs = []
        
        for qa in qa_pairs:
            question_lower = qa['question'].lower()
            if question_lower not in seen_questions:
                seen_questions.add(question_lower)
                unique_pairs.append(qa)
        
        return unique_pairs
    
    def _ensure_diversity(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保问题多样性"""
        if len(qa_pairs) <= 1:
            return qa_pairs
        
        # 按类型分组
        type_groups = {}
        for qa in qa_pairs:
            qtype = qa.get('question_type', 'unknown')
            if qtype not in type_groups:
                type_groups[qtype] = []
            type_groups[qtype].append(qa)
        
        # 从每个类型中选择
        diverse_pairs = []
        max_per_type = max(1, self.qa_config.num_questions // len(self.qa_config.question_types))
        
        for qtype in self.qa_config.question_types:
            if qtype in type_groups:
                diverse_pairs.extend(type_groups[qtype][:max_per_type])
        
        # 如果数量不够，补充其他类型
        if len(diverse_pairs) < self.qa_config.num_questions:
            remaining = self.qa_config.num_questions - len(diverse_pairs)
            for pairs in type_groups.values():
                for pair in pairs:
                    if pair not in diverse_pairs and remaining > 0:
                        diverse_pairs.append(pair)
                        remaining -= 1
        
        return diverse_pairs[:self.qa_config.num_questions]
    
    def _quality_check(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """质量检查"""
        valid_pairs = []
        
        for qa in qa_pairs:
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            
            # 长度检查
            if not (self.qa_config.min_question_length <= len(question) <= self.qa_config.max_question_length):
                continue
            
            if not (self.qa_config.min_answer_length <= len(answer) <= self.qa_config.max_answer_length):
                continue
            
            # 问题格式检查
            if not question.endswith('?') and not question.endswith('？'):
                qa['question'] = question + '?'
            
            # 置信度检查
            if qa.get('confidence', 1.0) < 0.5:
                continue
            
            valid_pairs.append(qa)
        
        return valid_pairs
    
    def _extract_text(self, item: Any) -> Optional[str]:
        """提取文本内容"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            return item.get('text') or item.get('content')
        elif hasattr(item, 'text'):
            return item.text
        else:
            return None