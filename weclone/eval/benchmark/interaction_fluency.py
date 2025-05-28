"""
Interaction Fluency Benchmark

Measures conversation flow and interruptions in human-AI interactions.
Evaluates how smoothly the conversation progresses without interruptions,
timeouts, or awkward pauses.
"""

from typing import Any, Dict, List
from .base import BaseBenchmark, BenchmarkResult
from ..types import JobContext


class InteractionFluencyBenchmark(BaseBenchmark):
    """Measures conversation flow and interruptions"""
    
    @property
    def _get_name(self) -> str:
        return "interaction_fluency"
    
    def required_artifacts(self) -> List[str]:
        return ["response_times", "turn_intervals"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute interaction fluency metrics
        
        Metrics computed:
        - interrupt_count: Number of user interruptions (quick follow-ups)
        - timeout_resend_count: Number of timeout-induced resends
        - avg_turn_interval: Average time between conversation turns
        """
        if not self.validate_artifacts(job_ctx):
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics={"interrupt_count": 0, "timeout_resend_count": 0, "avg_turn_interval": 0}
            )
        
        response_times = job_ctx.artifacts.get("response_times", [])
        turn_intervals = job_ctx.artifacts.get("turn_intervals", [])
        
        # Configuration parameters with defaults
        interrupt_threshold = self.config.get("interrupt_threshold_ms", 500)  # <0.5s = interrupt
        timeout_threshold = self.config.get("timeout_threshold_ms", 30000)   # >30s = timeout
        
        # Count interrupts and timeouts
        interrupt_count = sum(1 for interval in turn_intervals if interval < interrupt_threshold)
        timeout_resend_count = sum(1 for rt in response_times if rt > timeout_threshold)
        
        # Calculate average turn interval
        avg_turn_interval = sum(turn_intervals) / len(turn_intervals) if turn_intervals else 0
        
        metrics = {
            "interrupt_count": interrupt_count,
            "timeout_resend_count": timeout_resend_count,
            "avg_turn_interval": round(avg_turn_interval, 3)
        }
        
        metadata = {
            "interrupt_threshold_ms": interrupt_threshold,
            "timeout_threshold_ms": timeout_threshold,
            "total_turns": len(turn_intervals),
            "total_response_times": len(response_times)
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            metadata=metadata
        )
    
    def get_description(self) -> str:
        return "Measures conversation flow quality by detecting interruptions and timeouts"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "interrupt_threshold_ms": {
                    "type": "number",
                    "description": "Threshold in milliseconds below which a turn is considered an interrupt",
                    "default": 500,
                    "minimum": 0
                },
                "timeout_threshold_ms": {
                    "type": "number", 
                    "description": "Threshold in milliseconds above which a response is considered timed out",
                    "default": 30000,
                    "minimum": 1000
                }
            },
            "additionalProperties": False
        } 