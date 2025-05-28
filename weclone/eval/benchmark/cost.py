"""
Cost Benchmark

Measures API usage costs and resource efficiency for conversational AI.
Evaluates token usage, cost per conversation, and cost efficiency metrics.
"""

from typing import Any, Dict, List, Tuple
from .base import BaseBenchmark, BenchmarkResult
from ..types import JobContext


class CostBenchmark(BaseBenchmark):
    """Measures API usage costs"""
    
    @property
    def _get_name(self) -> str:
        return "cost"
    
    def required_artifacts(self) -> List[str]:
        return ["token_usage"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute cost metrics
        
        Metrics computed:
        - prompt_tokens: Number of prompt tokens used
        - completion_tokens: Number of completion tokens generated
        - usd_cost: Total cost in USD
        - cost_per_token: Cost per token (optional)
        """
        if not self.validate_artifacts(job_ctx):
            return BenchmarkResult(
                benchmark_name=self.name,
                metrics={"prompt_tokens": 0, "completion_tokens": 0, "usd_cost": 0.0}
            )
        
        token_usage = job_ctx.artifacts.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        # Get model-specific pricing
        model_name = job_ctx.model_config.name
        prompt_price, completion_price = self._get_model_pricing(model_name)
        
        # Calculate costs
        usd_cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000
        
        metrics = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd_cost": round(usd_cost, 6)
        }
        
        # Add cost efficiency metrics
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            metrics["cost_per_token"] = round(usd_cost / total_tokens, 8)
        
        if completion_tokens > 0:
            metrics["cost_per_completion_token"] = round(usd_cost / completion_tokens, 8)
        
        # Cost efficiency classification
        efficiency_class = self._classify_cost_efficiency(usd_cost, total_tokens)
        
        metadata = {
            "model_name": model_name,
            "prompt_price_per_1k": prompt_price,
            "completion_price_per_1k": completion_price,
            "total_tokens": total_tokens,
            "efficiency_class": efficiency_class,
            "conversation_id": job_ctx.conv_id
        }
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            metadata=metadata
        )
    
    def _get_model_pricing(self, model_name: str) -> Tuple[float, float]:
        """
        Get model-specific pricing (USD per 1K tokens)
        
        Returns:
            Tuple of (prompt_price, completion_price)
        """
        # Custom pricing from config
        custom_pricing = self.config.get("model_pricing", {})
        if model_name in custom_pricing:
            pricing = custom_pricing[model_name]
            return pricing.get("prompt", 0.001), pricing.get("completion", 0.002)
        
        # Default pricing based on model type
        model_lower = model_name.lower()
        if "gpt-4" in model_lower:
            if "turbo" in model_lower:
                return 0.01, 0.03  # GPT-4 Turbo
            else:
                return 0.03, 0.06  # GPT-4
        elif "gpt-3.5" in model_lower:
            return 0.001, 0.002  # GPT-3.5 Turbo
        elif "claude" in model_lower:
            if "3-opus" in model_lower:
                return 0.015, 0.075  # Claude 3 Opus
            elif "3-sonnet" in model_lower:
                return 0.003, 0.015  # Claude 3 Sonnet
            else:
                return 0.001, 0.005  # Claude 3 Haiku
        else:
            # Default for unknown models
            return self.config.get("default_prompt_price", 0.001), self.config.get("default_completion_price", 0.001)
    
    def _classify_cost_efficiency(self, usd_cost: float, total_tokens: int) -> str:
        """Classify cost efficiency into categories"""
        if total_tokens == 0:
            return "no_tokens"
        
        cost_per_token = usd_cost / total_tokens
        
        # Configurable thresholds
        excellent_threshold = self.config.get("excellent_cost_per_token", 0.00001)
        good_threshold = self.config.get("good_cost_per_token", 0.00005)
        acceptable_threshold = self.config.get("acceptable_cost_per_token", 0.0001)
        
        if cost_per_token <= excellent_threshold:
            return "excellent"
        elif cost_per_token <= good_threshold:
            return "good"
        elif cost_per_token <= acceptable_threshold:
            return "acceptable"
        else:
            return "expensive"
    
    def get_description(self) -> str:
        return "Measures API usage costs including token consumption and USD cost calculations"
    
    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_pricing": {
                    "type": "object",
                    "description": "Custom pricing for specific models",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "number", "minimum": 0},
                            "completion": {"type": "number", "minimum": 0}
                        }
                    }
                },
                "default_prompt_price": {
                    "type": "number",
                    "description": "Default prompt price per 1K tokens",
                    "default": 0.001,
                    "minimum": 0
                },
                "default_completion_price": {
                    "type": "number",
                    "description": "Default completion price per 1K tokens",
                    "default": 0.001,
                    "minimum": 0
                },
                "excellent_cost_per_token": {
                    "type": "number",
                    "description": "Threshold for excellent cost efficiency",
                    "default": 0.00001,
                    "minimum": 0
                },
                "good_cost_per_token": {
                    "type": "number",
                    "description": "Threshold for good cost efficiency",
                    "default": 0.00005,
                    "minimum": 0
                },
                "acceptable_cost_per_token": {
                    "type": "number",
                    "description": "Threshold for acceptable cost efficiency",
                    "default": 0.0001,
                    "minimum": 0
                }
            },
            "additionalProperties": False
        } 