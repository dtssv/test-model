"""
评测运行器
支持多模态模型评测
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """评测结果"""
    benchmark_name: str
    metrics: Dict[str, float]
    num_samples: int
    num_correct: int
    accuracy: float
    details: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "details": self.details,
        }


class BenchmarkRunner:
    """
    多模态模型评测运行器
    支持VQA、Caption、MMLU等评测集
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        output_dir: str = "./eval_results",
    ):
        """
        初始化评测运行器
        
        Args:
            model: 模型实例
            tokenizer: tokenizer实例
            device: 设备类型
            output_dir: 结果输出目录
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 注册评测集
        self.benchmarks = {
            "vqav2": self._eval_vqav2,
            "gqa": self._eval_gqa,
            "textvqa": self._eval_textvqa,
            "coco_caption": self._eval_coco_caption,
            "mmlu": self._eval_mmlu,
            "mmbench": self._eval_mmbench,
        }
    
    def run_benchmark(
        self,
        benchmark_name: str,
        data_path: str,
        split: str = "val",
        **kwargs
    ) -> BenchmarkResult:
        """
        运行指定评测集
        
        Args:
            benchmark_name: 评测集名称
            data_path: 数据路径
            split: 数据集分割
            **kwargs: 其他参数
        
        Returns:
            评测结果
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        self.logger.info(f"Running benchmark: {benchmark_name}")
        
        # 运行评测
        eval_func = self.benchmarks[benchmark_name]
        result = eval_func(data_path, split, **kwargs)
        
        # 保存结果
        self._save_result(result, benchmark_name)
        
        self.logger.info(f"Benchmark {benchmark_name} completed: accuracy={result.accuracy:.2%}")
        
        return result
    
    def run_all_benchmarks(
        self,
        data_dir: str,
        benchmarks: List[str] = None,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """
        运行所有评测集
        
        Args:
            data_dir: 数据目录
            benchmarks: 要运行的评测集列表，None表示全部
            **kwargs: 其他参数
        
        Returns:
            评测结果字典
        """
        if benchmarks is None:
            benchmarks = list(self.benchmarks.keys())
        
        results = {}
        for benchmark_name in benchmarks:
            try:
                data_path = Path(data_dir) / benchmark_name
                result = self.run_benchmark(benchmark_name, str(data_path), **kwargs)
                results[benchmark_name] = result
            except Exception as e:
                self.logger.error(f"Error running benchmark {benchmark_name}: {e}")
        
        return results
    
    def _eval_vqav2(self, data_path: str, split: str = "val", **kwargs) -> BenchmarkResult:
        """VQAv2评测"""
        self.logger.info("Evaluating VQAv2...")
        
        # 加载数据
        questions, answers, images = self._load_vqa_data(data_path, split)
        
        num_correct = 0
        total = len(questions)
        details = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (question, answer_list, image) in enumerate(tqdm(zip(questions, answers, images), total=total)):
                # 生成答案
                predicted_answer = self._generate_answer(question, image)
                
                # 计算准确率（VQAv2使用多答案评分）
                score = self._compute_vqa_score(predicted_answer, answer_list)
                num_correct += score
                
                if i < 100:  # 只保存前100个样本的详细信息
                    details.append({
                        "question": question,
                        "predicted": predicted_answer,
                        "ground_truth": answer_list,
                        "score": score,
                    })
        
        accuracy = num_correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name="vqav2",
            metrics={"accuracy": accuracy},
            num_samples=total,
            num_correct=int(num_correct),
            accuracy=accuracy,
            details=details,
        )
    
    def _eval_gqa(self, data_path: str, split: str = "val", **kwargs) -> BenchmarkResult:
        """GQA评测"""
        self.logger.info("Evaluating GQA...")
        
        # 加载数据
        questions, answers, images = self._load_gqa_data(data_path, split)
        
        num_correct = 0
        total = len(questions)
        details = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (question, answer, image) in enumerate(tqdm(zip(questions, answers, images), total=total)):
                # 生成答案
                predicted_answer = self._generate_answer(question, image)
                
                # 计算准确率
                is_correct = self._compute_accuracy(predicted_answer, [answer])
                num_correct += int(is_correct)
                
                if i < 100:
                    details.append({
                        "question": question,
                        "predicted": predicted_answer,
                        "ground_truth": answer,
                        "correct": is_correct,
                    })
        
        accuracy = num_correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name="gqa",
            metrics={"accuracy": accuracy},
            num_samples=total,
            num_correct=num_correct,
            accuracy=accuracy,
            details=details,
        )
    
    def _eval_textvqa(self, data_path: str, split: str = "val", **kwargs) -> BenchmarkResult:
        """TextVQA评测"""
        self.logger.info("Evaluating TextVQA...")
        
        # 加载数据
        questions, answers, images = self._load_textvqa_data(data_path, split)
        
        num_correct = 0
        total = len(questions)
        
        self.model.eval()
        with torch.no_grad():
            for question, answer_list, image in tqdm(zip(questions, answers, images), total=total):
                # 生成答案
                predicted_answer = self._generate_answer(question, image)
                
                # 计算准确率
                score = self._compute_vqa_score(predicted_answer, answer_list)
                num_correct += score
        
        accuracy = num_correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name="textvqa",
            metrics={"accuracy": accuracy},
            num_samples=total,
            num_correct=int(num_correct),
            accuracy=accuracy,
        )
    
    def _eval_coco_caption(self, data_path: str, split: str = "val", **kwargs) -> BenchmarkResult:
        """COCO Caption评测"""
        self.logger.info("Evaluating COCO Caption...")
        
        # 加载数据
        images, references = self._load_caption_data(data_path, split)
        
        predictions = []
        total = len(images)
        
        self.model.eval()
        with torch.no_grad():
            for image in tqdm(images, total=total):
                # 生成描述
                caption = self._generate_caption(image)
                predictions.append(caption)
        
        # 计算指标
        metrics = self._compute_caption_metrics(predictions, references)
        
        return BenchmarkResult(
            benchmark_name="coco_caption",
            metrics=metrics,
            num_samples=total,
            num_correct=0,
            accuracy=metrics.get("cider", 0.0),
        )
    
    def _eval_mmlu(self, data_path: str, split: str = "test", **kwargs) -> BenchmarkResult:
        """MMLU评测"""
        self.logger.info("Evaluating MMLU...")
        
        # 加载数据
        questions, choices, answers = self._load_mmlu_data(data_path, split)
        
        num_correct = 0
        total = len(questions)
        
        self.model.eval()
        with torch.no_grad():
            for question, choice_list, answer in tqdm(zip(questions, choices, answers), total=total):
                # 构建prompt
                prompt = self._format_mmlu_question(question, choice_list)
                
                # 生成答案
                predicted = self._generate_text(prompt)
                
                # 解析答案
                predicted_answer = self._parse_mmlu_answer(predicted)
                
                # 检查是否正确
                if predicted_answer == answer:
                    num_correct += 1
        
        accuracy = num_correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name="mmlu",
            metrics={"accuracy": accuracy},
            num_samples=total,
            num_correct=num_correct,
            accuracy=accuracy,
        )
    
    def _eval_mmbench(self, data_path: str, split: str = "val", **kwargs) -> BenchmarkResult:
        """MMBench评测"""
        self.logger.info("Evaluating MMBench...")
        
        # 加载数据
        questions, choices, answers, images = self._load_mmbench_data(data_path, split)
        
        num_correct = 0
        total = len(questions)
        
        self.model.eval()
        with torch.no_grad():
            for question, choice_list, answer, image in tqdm(zip(questions, choices, answers, images), total=total):
                # 生成答案
                predicted = self._generate_answer(question, image)
                
                # 解析答案
                predicted_answer = self._parse_choice(predicted, choice_list)
                
                # 检查是否正确
                if predicted_answer == answer:
                    num_correct += 1
        
        accuracy = num_correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            benchmark_name="mmbench",
            metrics={"accuracy": accuracy},
            num_samples=total,
            num_correct=num_correct,
            accuracy=accuracy,
        )
    
    def _generate_answer(self, question: str, image) -> str:
        """生成答案（VQA任务）"""
        # 构建prompt
        prompt = f"<image>\nQuestion: {question}\nAnswer:"
        
        # 模型推理
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        
        # 解码
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def _generate_caption(self, image) -> str:
        """生成描述（Caption任务）"""
        prompt = "<image>\nDescribe this image:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Describe this image:" in caption:
            caption = caption.split("Describe this image:")[-1].strip()
        
        return caption
    
    def _generate_text(self, prompt: str) -> str:
        """生成文本（纯文本任务）"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _compute_vqa_score(self, predicted: str, answers: List[str]) -> float:
        """计算VQA评分"""
        predicted = predicted.lower().strip()
        scores = []
        for answer in answers:
            answer = answer.lower().strip()
            if predicted == answer:
                scores.append(1.0)
            elif predicted in answer or answer in predicted:
                scores.append(0.5)
            else:
                scores.append(0.0)
        
        return min(1.0, sum(scores) / len(answers) * 3.0)
    
    def _compute_accuracy(self, predicted: str, answers: List[str]) -> bool:
        """计算准确率"""
        predicted = predicted.lower().strip()
        for answer in answers:
            if predicted == answer.lower().strip():
                return True
        return False
    
    def _compute_caption_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """计算Caption指标"""
        try:
            from pycocoevalcap.cider.cider import Cider
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            
            # 准备数据
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [pred] for i, pred in enumerate(predictions)}
            
            # 计算CIDEr
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            
            # 计算BLEU
            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts, res)
            
            # 计算METEOR
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
            
            # 计算ROUGE
            rouge_scorer = Rouge()
            rouge_score, _ = rouge_scorer.compute_score(gts, res)
            
            return {
                "cider": cider_score,
                "bleu1": bleu_scores[0],
                "bleu4": bleu_scores[3],
                "meteor": meteor_score,
                "rouge": rouge_score,
            }
        except ImportError:
            self.logger.warning("pycocoevalcap not installed, skipping caption metrics")
            return {}
    
    def _format_mmlu_question(self, question: str, choices: List[str]) -> str:
        """格式化MMLU问题"""
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        return prompt
    
    def _parse_mmlu_answer(self, answer: str) -> str:
        """解析MMLU答案"""
        answer = answer.strip().upper()
        if answer and answer[0] in "ABCD":
            return answer[0]
        return ""
    
    def _parse_choice(self, answer: str, choices: List[str]) -> str:
        """解析选择题答案"""
        answer = answer.strip().upper()
        if answer and answer[0] in "ABCD"[:len(choices)]:
            return answer[0]
        return ""
    
    def _load_vqa_data(self, data_path: str, split: str):
        """加载VQA数据"""
        # TODO: 实现数据加载
        return [], [], []
    
    def _load_gqa_data(self, data_path: str, split: str):
        """加载GQA数据"""
        # TODO: 实现数据加载
        return [], [], []
    
    def _load_textvqa_data(self, data_path: str, split: str):
        """加载TextVQA数据"""
        # TODO: 实现数据加载
        return [], [], []
    
    def _load_caption_data(self, data_path: str, split: str):
        """加载Caption数据"""
        # TODO: 实现数据加载
        return [], []
    
    def _load_mmlu_data(self, data_path: str, split: str):
        """加载MMLU数据"""
        # TODO: 实现数据加载
        return [], [], []
    
    def _load_mmbench_data(self, data_path: str, split: str):
        """加载MMBench数据"""
        # TODO: 实现数据加载
        return [], [], [], []
    
    def _save_result(self, result: BenchmarkResult, benchmark_name: str):
        """保存评测结果"""
        result_file = self.output_dir / f"{benchmark_name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Result saved to {result_file}")
    
    def generate_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """生成评测报告"""
        report_lines = [
            "# Evaluation Report\n",
            f"Total benchmarks: {len(results)}\n\n",
        ]
        
        for benchmark_name, result in results.items():
            report_lines.append(f"## {benchmark_name}\n")
            report_lines.append(f"- Accuracy: {result.accuracy:.2%}\n")
            report_lines.append(f"- Samples: {result.num_samples}\n")
            report_lines.append(f"- Correct: {result.num_correct}\n")
            
            if result.metrics:
                report_lines.append("- Metrics:\n")
                for metric_name, metric_value in result.metrics.items():
                    report_lines.append(f"  - {metric_name}: {metric_value:.4f}\n")
            
            report_lines.append("\n")
        
        report = "".join(report_lines)
        
        # 保存报告
        report_file = self.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {report_file}")
        
        return report