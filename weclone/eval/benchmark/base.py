"""
Base classes for benchmark implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
from ..types import JobContext


@dataclass
class BenchmarkResult:
    """Standard result format for benchmarks"""
    benchmark_name: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseBenchmark(ABC):
    """Base class for all benchmark implementations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize benchmark with optional configuration
        
        Args:
            config: Benchmark-specific configuration parameters
        """
        self.config = config or {}
        self.name = self._get_name
    
    @property
    @abstractmethod
    def _get_name(self) -> str:
        """Return the benchmark name"""
        pass
    
    @abstractmethod
    def required_artifacts(self) -> List[str]:
        """
        Return list of required artifacts for this benchmark
        
        Returns:
            List of artifact keys that must be present in JobContext
        """
        pass
    
    @abstractmethod
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        """
        Compute benchmark metrics from job context
        
        Args:
            job_ctx: Job context containing conversation data and artifacts
            
        Returns:
            BenchmarkResult with computed metrics
        """
        pass
    
    def validate_artifacts(self, job_ctx: JobContext) -> bool:
        """
        Validate that all required artifacts are present
        
        Args:
            job_ctx: Job context to validate
            
        Returns:
            True if all required artifacts are present, False otherwise
        """
        required = set(self.required_artifacts())
        available = set(job_ctx.artifacts.keys())
        missing = required - available
        
        if missing:
            from weclone.utils.log import logger
            logger.warning(f"Benchmark {self.name} missing artifacts: {missing}")
            return False
        
        return True
    
    def get_description(self) -> str:
        """
        Get human-readable description of the benchmark
        
        Returns:
            Description string
        """
        return f"{self.name} benchmark"
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for this benchmark
        
        Returns:
            JSON schema describing valid configuration options
        """
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        } 