"""
WeClone Evaluation Benchmarks

This package contains individual benchmark implementations that can be used
to evaluate conversational AI models across different dimensions.
"""

from .base import BaseBenchmark, BenchmarkResult
from .latency import LatencyBenchmark
from .cost import CostBenchmark
from .chathumanscore import ChatHumanScore

__all__ = [
    'BaseBenchmark',
    'BenchmarkResult',
    'LatencyBenchmark',
    'CostBenchmark',
    'ChatHumanScore'
]

# Registry for easy benchmark discovery
AVAILABLE_BENCHMARKS = {
    'latency': LatencyBenchmark,
    'cost': CostBenchmark,
    'chathumanscore': ChatHumanScore
} 