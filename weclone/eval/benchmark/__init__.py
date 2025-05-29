"""
WeClone Evaluation Benchmarks

This package contains individual benchmark implementations that can be used
to evaluate conversational AI models across different dimensions.
"""

from .base import BaseBenchmark, BenchmarkResult
from .interaction_fluency import InteractionFluencyBenchmark
from .sentiment_satisfaction import SentimentSatisfactionBenchmark
from .task_success import TaskSuccessBenchmark
from .latency import LatencyBenchmark
from .cost import CostBenchmark
from .chathumanscore import ChatHumanScore

__all__ = [
    'BaseBenchmark',
    'BenchmarkResult',
    'InteractionFluencyBenchmark',
    'SentimentSatisfactionBenchmark', 
    'TaskSuccessBenchmark',
    'LatencyBenchmark',
    'CostBenchmark',
    'ChatHumanScore'
]

# Registry for easy benchmark discovery
AVAILABLE_BENCHMARKS = {
    'interaction_fluency': InteractionFluencyBenchmark,
    'sentiment_satisfaction': SentimentSatisfactionBenchmark,
    'task_success': TaskSuccessBenchmark,
    'latency': LatencyBenchmark,
    'cost': CostBenchmark,
    'chathumanscore': ChatHumanScore
} 