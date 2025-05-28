"""
Type definitions for WeClone evaluation framework
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    params: Dict[str, Any]
    host: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class PromptConfig:
    """Prompt configuration"""
    id: str
    version: str
    content: Optional[str] = None
    file: Optional[str] = None


@dataclass
class TestCase:
    """Individual test case"""
    conv_id: str
    turns: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]


@dataclass
class JobContext:
    """Context for metric computation"""
    run_id: str
    conv_id: str
    model_config: ModelConfig
    prompt_config: PromptConfig
    test_case: TestCase
    response_data: Dict[str, Any]
    artifacts: Dict[str, Any]  # Additional data for metrics 