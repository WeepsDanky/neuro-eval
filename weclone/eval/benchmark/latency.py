"""
Latency Benchmark

Measures response timing and performance characteristics of conversational AI.
Evaluates first token latency, full response time, and other timing metrics.
"""

from typing import Any, Dict, List
from .base import BaseBenchmark, BenchmarkResult
from ..types import JobContext


class LatencyBenchmark(BaseBenchmark):
    """Measures response timing"""
    
    @property
    def _get_name(self) -> str:
        return "latency"
    
    def required_artifacts(self) -> List[str]:
        return ["first_token_time", "full_response_time"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute latency metrics
        
        Metrics computed:
        - first_token_ms: Time to first token in milliseconds
        - full_response_ms: Time to complete response in milliseconds
        - throughput_tokens_per_sec: Token generation throughput (optional)
        """
        if not self.validate_artifacts(job_ctx):
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics={"first_token_ms": 0, "full_response_ms": 0}
            )
        
        first_token_ms = job_ctx.artifacts.get("first_token_time", 0)
        full_response_ms = job_ctx.artifacts.get("full_response_time", 0)
        
        # Optional: Calculate throughput if token info is available
        token_usage = job_ctx.artifacts.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        metrics = {
            "first_token_ms": round(first_token_ms, 2),
            "full_response_ms": round(full_response_ms, 2)
        }
        
        # Add throughput if we have token data
        if completion_tokens > 0 and full_response_ms > 0:
            # Convert ms to seconds for throughput calculation
            throughput = completion_tokens / (full_response_ms / 1000.0)
            metrics["throughput_tokens_per_sec"] = round(throughput, 2)
        
        # Performance classification
        performance_class = self._classify_performance(first_token_ms, full_response_ms)
        
        metadata = {
            "completion_tokens": completion_tokens,
            "performance_class": performance_class,
            "first_token_threshold_ms": self.config.get("first_token_threshold_ms", 1000),
            "full_response_threshold_ms": self.config.get("full_response_threshold_ms", 5000)
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            metadata=metadata
        )
    
    def _classify_performance(self, first_token_ms: float, full_response_ms: float) -> str:
        """Classify response performance into categories"""
        # Configurable thresholds
        first_token_excellent = self.config.get("first_token_excellent_ms", 500)
        first_token_good = self.config.get("first_token_good_ms", 1000)
        full_response_excellent = self.config.get("full_response_excellent_ms", 2000)
        full_response_good = self.config.get("full_response_good_ms", 5000)
        
        # Classify based on both metrics
        if first_token_ms <= first_token_excellent and full_response_ms <= full_response_excellent:
            return "excellent"
        elif first_token_ms <= first_token_good and full_response_ms <= full_response_good:
            return "good"
        elif full_response_ms <= full_response_good * 2:
            return "acceptable"
        else:
            return "poor"
    
    def get_description(self) -> str:
        return "Measures response timing including first token latency and full response time"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "first_token_threshold_ms": {
                    "type": "number",
                    "description": "Threshold for acceptable first token latency",
                    "default": 1000,
                    "minimum": 100
                },
                "full_response_threshold_ms": {
                    "type": "number",
                    "description": "Threshold for acceptable full response time",
                    "default": 5000,
                    "minimum": 500
                },
                "first_token_excellent_ms": {
                    "type": "number",
                    "description": "Threshold for excellent first token latency",
                    "default": 500,
                    "minimum": 50
                },
                "first_token_good_ms": {
                    "type": "number",
                    "description": "Threshold for good first token latency",
                    "default": 1000,
                    "minimum": 100
                },
                "full_response_excellent_ms": {
                    "type": "number",
                    "description": "Threshold for excellent full response time",
                    "default": 2000,
                    "minimum": 200
                },
                "full_response_good_ms": {
                    "type": "number",
                    "description": "Threshold for good full response time",
                    "default": 5000,
                    "minimum": 500
                }
            },
            "additionalProperties": False
        } 