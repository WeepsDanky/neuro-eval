"""
Task Success Benchmark

Measures task completion accuracy and quality for conversational AI.
Evaluates how well the AI performs specific tasks through various metrics
like retrieval precision, generation quality, and function call accuracy.
"""

from typing import Any, Dict, List
from .base import BaseBenchmark, BenchmarkResult
from ..types import JobContext


class TaskSuccessBenchmark(BaseBenchmark):
    """Measures task completion accuracy"""
    
    @property
    def _get_name(self) -> str:
        return "task_success"
    
    def required_artifacts(self) -> List[str]:
        return ["expected_output", "actual_output"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute task success metrics
        
        Metrics computed:
        - retrieval_precision: Precision of information retrieval
        - gen_bleu: BLEU-like score for generation quality
        - function_call_accuracy: Accuracy of function calls
        """
        if not self.validate_artifacts(job_ctx):
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics={"retrieval_precision": 0.0, "gen_bleu": 0.0, "function_call_accuracy": 0.0}
            )
        
        expected = job_ctx.artifacts.get("expected_output", "")
        actual = job_ctx.artifacts.get("actual_output", "")
        
        if not expected or not actual:
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics={"retrieval_precision": 0.0, "gen_bleu": 0.0, "function_call_accuracy": 0.0}
            )
        
        # Compute retrieval precision (word overlap)
        retrieval_precision = self._compute_retrieval_precision(expected, actual)
        
        # Compute generation BLEU score
        gen_bleu = self._compute_generation_bleu(expected, actual)
        
        # Compute function call accuracy
        function_call_accuracy = self._compute_function_accuracy(expected, actual)
        
        metrics = {
            "retrieval_precision": round(retrieval_precision, 3),
            "gen_bleu": round(gen_bleu, 3),
            "function_call_accuracy": round(function_call_accuracy, 3)
        }
        
        metadata = {
            "expected_length": len(expected),
            "actual_length": len(actual),
            "word_overlap_method": "simple_intersection",
            "bleu_method": "containment_based",
            "function_keywords": self.config.get("function_keywords", ["函数", "调用", "执行"])
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            metadata=metadata
        )
    
    def _compute_retrieval_precision(self, expected: str, actual: str) -> float:
        """Compute word-level retrieval precision"""
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words & actual_words)
        precision = overlap / len(expected_words)
        return precision
    
    def _compute_generation_bleu(self, expected: str, actual: str) -> float:
        """Compute a simplified BLEU-like score"""
        # Exact match bonus
        if actual.lower().strip() == expected.lower().strip():
            return 1.0
        
        # Substring containment
        bleu_threshold = self.config.get("bleu_threshold", 0.8)
        high_bleu_score = self.config.get("high_bleu_score", 0.8)
        low_bleu_score = self.config.get("low_bleu_score", 0.3)
        
        if (actual.lower() in expected.lower() or expected.lower() in actual.lower()):
            return high_bleu_score
        
        # Word overlap based scoring
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap_ratio = len(expected_words & actual_words) / len(expected_words)
        if overlap_ratio >= bleu_threshold:
            return high_bleu_score
        else:
            return low_bleu_score
    
    def _compute_function_accuracy(self, expected: str, actual: str) -> float:
        """Compute function call accuracy"""
        function_keywords = self.config.get("function_keywords", ["函数", "调用", "执行", "方法", "API"])
        
        expected_has_function = any(keyword in expected for keyword in function_keywords)
        actual_has_function = any(keyword in actual for keyword in function_keywords)
        
        # Both have function references
        if expected_has_function and actual_has_function:
            return 1.0
        # Neither have function references
        elif not expected_has_function and not actual_has_function:
            return 1.0
        # Mismatch
        else:
            return 0.0
    
    def get_description(self) -> str:
        return "Evaluates task completion quality through retrieval precision, generation quality, and function accuracy"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bleu_threshold": {
                    "type": "number",
                    "description": "Word overlap threshold for high BLEU scores",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "high_bleu_score": {
                    "type": "number",
                    "description": "BLEU score for high-quality matches",
                    "default": 0.8,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "low_bleu_score": {
                    "type": "number",
                    "description": "BLEU score for low-quality matches",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "function_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords that indicate function calls",
                    "default": ["函数", "调用", "执行", "方法", "API"]
                }
            },
            "additionalProperties": False
        } 