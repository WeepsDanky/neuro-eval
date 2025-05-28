"""
WeClone Evaluation Framework

A robust evaluation system for testing conversational AI models with:
- Multi-metric benchmarking (Interaction Fluency, Sentiment Satisfaction, Task Success, Latency, Cost)
- Plugin architecture for extensible metrics
- YAML/JSON configuration support
- Comprehensive data persistence
- Multi-model and prompt variant testing
"""

import asyncio
import csv
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
import uuid

try:
    import yaml
except ImportError:
    yaml = None

from openai import OpenAI
from tqdm import tqdm

from weclone.utils.log import logger
from weclone.utils.config import load_config


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


class MetricPlugin(Protocol):
    """Protocol for metric plugins"""
    name: str
    
    def required_artifacts(self) -> List[str]:
        """Return list of required artifacts for this metric"""
        ...
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        """Compute metric scores from job context"""
        ...


class BaseMetric(ABC):
    """Base class for metric implementations"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def required_artifacts(self) -> List[str]:
        pass
    
    @abstractmethod
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        pass


class InteractionFluencyMetric(BaseMetric):
    """Measures conversation flow and interruptions"""
    
    @property
    def name(self) -> str:
        return "interaction_fluency"
    
    def required_artifacts(self) -> List[str]:
        return ["response_times", "turn_intervals"]
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        # Simulate interrupt detection and timeout handling
        response_times = job_ctx.artifacts.get("response_times", [])
        turn_intervals = job_ctx.artifacts.get("turn_intervals", [])
        
        interrupt_count = sum(1 for interval in turn_intervals if interval < 0.5)  # <0.5s = interrupt
        timeout_resend_count = sum(1 for rt in response_times if rt > 30000)  # >30s = timeout
        
        return {
            "interrupt_count": interrupt_count,
            "timeout_resend_count": timeout_resend_count,
            "avg_turn_interval": sum(turn_intervals) / len(turn_intervals) if turn_intervals else 0
        }


class SentimentSatisfactionMetric(BaseMetric):
    """Measures user satisfaction through sentiment analysis"""
    
    @property
    def name(self) -> str:
        return "sentiment_satisfaction"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text"]
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        # Simplified sentiment scoring - in practice, use actual sentiment analysis
        conversation_text = job_ctx.artifacts.get("conversation_text", "")
        
        # Mock sentiment analysis
        positive_words = ["好", "棒", "满意", "喜欢", "不错", "很好"]
        negative_words = ["差", "糟糕", "不满", "讨厌", "不好", "失望"]
        
        pos_count = sum(word in conversation_text for word in positive_words)
        neg_count = sum(word in conversation_text for word in negative_words)
        
        # Scale 1-5
        if pos_count > neg_count:
            rating = min(5, 3 + (pos_count - neg_count) * 0.5)
        else:
            rating = max(1, 3 - (neg_count - pos_count) * 0.5)
        
        return {
            "post_chat_rating": round(rating, 2),
            "sentiment_score": (pos_count - neg_count) / max(1, pos_count + neg_count)
        }


class TaskSuccessMetric(BaseMetric):
    """Measures task completion accuracy"""
    
    @property
    def name(self) -> str:
        return "task_success"
    
    def required_artifacts(self) -> List[str]:
        return ["expected_output", "actual_output"]
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        expected = job_ctx.artifacts.get("expected_output", "")
        actual = job_ctx.artifacts.get("actual_output", "")
        
        # Simple string matching - in practice, use BLEU/ROUGE
        if not expected or not actual:
            return {"retrieval_precision": 0.0, "gen_bleu": 0.0, "function_call_accuracy": 0.0}
        
        # Mock metrics
        retrieval_precision = len(set(expected.split()) & set(actual.split())) / len(set(expected.split()))
        gen_bleu = 0.8 if actual.lower() in expected.lower() or expected.lower() in actual.lower() else 0.3
        function_call_accuracy = 1.0 if "函数" in actual and "函数" in expected else 0.0
        
        return {
            "retrieval_precision": round(retrieval_precision, 3),
            "gen_bleu": round(gen_bleu, 3),
            "function_call_accuracy": round(function_call_accuracy, 3)
        }


class LatencyMetric(BaseMetric):
    """Measures response timing"""
    
    @property
    def name(self) -> str:
        return "latency"
    
    def required_artifacts(self) -> List[str]:
        return ["first_token_time", "full_response_time"]
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        first_token_ms = job_ctx.artifacts.get("first_token_time", 0)
        full_response_ms = job_ctx.artifacts.get("full_response_time", 0)
        
        return {
            "first_token_ms": first_token_ms,
            "full_response_ms": full_response_ms
        }


class CostMetric(BaseMetric):
    """Measures API usage costs"""
    
    @property
    def name(self) -> str:
        return "cost"
    
    def required_artifacts(self) -> List[str]:
        return ["token_usage"]
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        token_usage = job_ctx.artifacts.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        # Mock pricing (USD per 1K tokens)
        model_name = job_ctx.model_config.name
        if "gpt-4" in model_name.lower():
            prompt_price, completion_price = 0.03, 0.06
        elif "gpt-3.5" in model_name.lower():
            prompt_price, completion_price = 0.001, 0.002
        else:
            prompt_price, completion_price = 0.001, 0.001
        
        usd_cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1000
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd_cost": round(usd_cost, 6)
        }


class EvaluationFramework:
    """Main evaluation framework"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics = self._initialize_metrics()
        self.run_id = self._generate_run_id()
        self.output_dir = Path("eval_runs") / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration from YAML/JSON"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required for YAML configuration files. "
                        "Install it with: uv add pyyaml"
                    )
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _initialize_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize metric plugins"""
        available_metrics = {
            "interaction_fluency": InteractionFluencyMetric(),
            "sentiment_satisfaction": SentimentSatisfactionMetric(),
            "task_success": TaskSuccessMetric(),
            "latency": LatencyMetric(),
            "cost": CostMetric()
        }
        
        enabled_metrics = self.config.get("metrics", [])
        return {name: metric for name, metric in available_metrics.items() if name in enabled_metrics}
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"
    
    def _load_test_cases(self) -> List[TestCase]:
        """Load test cases from configuration"""
        test_cases = []
        cases_config = self.config.get("cases", [])
        
        for case_config in cases_config:
            if isinstance(case_config, dict) and "file" in case_config:
                cases_file = Path(case_config["file"])
                if cases_file.suffix == ".jsonl":
                    test_cases.extend(self._load_jsonl_cases(cases_file))
                elif cases_file.suffix == ".json":
                    test_cases.extend(self._load_json_cases(cases_file))
        
        # Support debug mode with limited test cases
        debug_limit = self.config.get("debug", {}).get("max_cases", None)
        if debug_limit and debug_limit > 0:
            test_cases = test_cases[:debug_limit]
            logger.info(f"Debug mode: Limited to {len(test_cases)} test cases")
        
        return test_cases
    
    def _load_jsonl_cases(self, file_path: Path) -> List[TestCase]:
        """Load test cases from JSONL file"""
        cases = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                turns = []
                for turn in data.get("conversation", []):
                    turns.append({"role": turn["role"], "content": turn["content"]})
                cases.append(TestCase(conv_id=str(i), turns=turns))
        return cases
    
    def _load_json_cases(self, file_path: Path) -> List[TestCase]:
        """Load test cases from JSON file (compatible with existing test_data.json)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cases = []
        questions = data.get("questions", [])
        for i, question_set in enumerate(questions):
            turns = []
            # Add all user inputs first
            for question in question_set:
                turns.append({"role": "user", "content": question})
            # Add one assistant response placeholder at the end
            turns.append({"role": "assistant", "content": ""})
            cases.append(TestCase(conv_id=str(i), turns=turns))
        
        return cases
    
    async def _run_conversation(self, model_config: ModelConfig, prompt_config: PromptConfig, test_case: TestCase) -> Dict[str, Any]:
        """Run a single conversation with timing and token tracking"""
        start_time = time.time()
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=model_config.api_key or "sk-test",
            base_url=model_config.host or "http://127.0.0.1:8005/v1"
        )
        
        messages = []
        if prompt_config.content:
            messages.append({"role": "system", "content": prompt_config.content})
        
        response_data = {
            "responses": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "timings": []
        }
        
        # Collect all user inputs and merge them for API call
        user_inputs = []
        for turn in test_case.turns:
            if turn["role"] == "user":
                user_inputs.append(turn["content"])
        
        # Only make one API call with merged user inputs (to comply with u/a alternating format)
        if user_inputs:
            logger.debug(f"Collected {len(user_inputs)} user inputs for conversation {test_case.conv_id}")
            
            # Merge all user inputs into one message
            merged_input = "\n".join(user_inputs)
            messages.append({"role": "user", "content": merged_input})
            
            turn_start = time.time()
            try:
                response = client.chat.completions.create(
                    model=model_config.params.get("model", "gpt-3.5-turbo"),
                    messages=messages,
                    **{k: v for k, v in model_config.params.items() if k != "model"}
                )
                turn_end = time.time()
                
                assistant_content = response.choices[0].message.content
                messages.append({"role": "assistant", "content": assistant_content})
                
                response_data["responses"].append(assistant_content)
                response_data["timings"].append((turn_end - turn_start) * 1000)  # ms
                
                # Track token usage
                if hasattr(response, 'usage'):
                    response_data["token_usage"]["prompt_tokens"] = response.usage.prompt_tokens
                    response_data["token_usage"]["completion_tokens"] = response.usage.completion_tokens
                
                logger.debug(f"Generated response for conversation {test_case.conv_id}: {len(assistant_content)} chars")
            
            except Exception as e:
                logger.error(f"Error in conversation {test_case.conv_id}: {e}")
                response_data["responses"].append(f"Error: {str(e)}")
                response_data["timings"].append(0)
        
        total_time = time.time() - start_time
        response_data["total_time_ms"] = total_time * 1000
        
        return response_data
    
    def _create_job_context(self, model_config: ModelConfig, prompt_config: PromptConfig, 
                           test_case: TestCase, response_data: Dict[str, Any]) -> JobContext:
        """Create job context for metric computation"""
        # Prepare artifacts
        artifacts = {
            "response_times": response_data.get("timings", []),
            "turn_intervals": [1.0] * len(response_data.get("timings", [])),  # Mock intervals
            "conversation_text": " ".join(response_data.get("responses", [])),
            "expected_output": "预期输出",  # Mock expected output
            "actual_output": " ".join(response_data.get("responses", [])),
            "first_token_time": response_data.get("timings", [0])[0] if response_data.get("timings") else 0,
            "full_response_time": response_data.get("total_time_ms", 0),
            "token_usage": response_data.get("token_usage", {})
        }
        
        return JobContext(
            run_id=self.run_id,
            conv_id=test_case.conv_id,
            model_config=model_config,
            prompt_config=prompt_config,
            test_case=test_case,
            response_data=response_data,
            artifacts=artifacts
        )
    
    def _save_run_metadata(self):
        """Save run metadata"""
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "enabled_metrics": list(self.metrics.keys())
        }
        
        with open(self.output_dir / "run_meta.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _save_dataset_csv(self, test_cases: List[TestCase]):
        """Save dataset in CSV format"""
        with open(self.output_dir / "dataset.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["conv_id", "turn_idx", "role", "content"])
            
            total_turns = 0
            assistant_responses = 0
            user_inputs = 0
            
            for case in test_cases:
                for turn_idx, turn in enumerate(case.turns):
                    writer.writerow([case.conv_id, turn_idx, turn["role"], turn["content"]])
                    total_turns += 1
                    if turn["role"] == "assistant" and turn["content"]:
                        assistant_responses += 1
                    elif turn["role"] == "user":
                        user_inputs += 1
            
            logger.info(f"Dataset saved: {len(test_cases)} conversations, {total_turns} total turns")
            logger.info(f"  - User inputs: {user_inputs}")
            logger.info(f"  - Assistant responses: {assistant_responses}")
            logger.info(f"  - Empty assistant responses: {total_turns - user_inputs - assistant_responses}")
    
    def _save_metrics_csv(self, results: List[Dict[str, Any]]):
        """Save metrics results in CSV format"""
        with open(self.output_dir / "metrics.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "conv_id", "model", "prompt", "metric", "value"])
            
            for result in results:
                for metric_name, scores in result["metrics"].items():
                    for score_name, score_value in scores.items():
                        writer.writerow([
                            self.run_id,
                            result["conv_id"],
                            result["model"],
                            result["prompt"],
                            f"{metric_name}.{score_name}",
                            score_value
                        ])
    
    def _save_latency_cost_csv(self, results: List[Dict[str, Any]]):
        """Save latency and cost data in CSV format"""
        with open(self.output_dir / "latency_cost.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "conv_id", "model", "n_tokens_prompt", "n_tokens_completion", "latency_ms", "cost_usd"])
            
            for result in results:
                latency_data = result["metrics"].get("latency", {})
                cost_data = result["metrics"].get("cost", {})
                
                writer.writerow([
                    self.run_id,
                    result["conv_id"],
                    result["model"],
                    cost_data.get("prompt_tokens", 0),
                    cost_data.get("completion_tokens", 0),
                    latency_data.get("full_response_ms", 0),
                    cost_data.get("usd_cost", 0)
                ])
    
    async def run_evaluation(self):
        """Run the complete evaluation"""
        logger.info(f"Starting evaluation run: {self.run_id}")
        
        # Load test cases
        test_cases = self._load_test_cases()
        logger.info(f"Loaded {len(test_cases)} test cases")
        
        # Save metadata and initial dataset (before responses)
        self._save_run_metadata()
        logger.debug("Saved run metadata")
        
        # Prepare model and prompt configurations
        models = [ModelConfig(**m) for m in self.config.get("models", [])]
        prompts = [PromptConfig(**p) for p in self.config.get("prompts", [])]
        
        # Run evaluations
        results = []
        completed_test_cases = []  # Store test cases with responses
        total_combinations = len(models) * len(prompts) * len(test_cases)
        
        with tqdm(total=total_combinations, desc="Running evaluations") as pbar:
            for model_config in models:
                for prompt_config in prompts:
                    for test_case in test_cases:
                        try:
                            # Run conversation
                            logger.debug(f"Running conversation for case {test_case.conv_id}")
                            response_data = await self._run_conversation(model_config, prompt_config, test_case)
                            
                            # Update test case with model response
                            updated_test_case = self._update_test_case_with_response(test_case, response_data)
                            completed_test_cases.append(updated_test_case)
                            
                            # Create job context
                            job_ctx = self._create_job_context(model_config, prompt_config, updated_test_case, response_data)
                            
                            # Compute metrics
                            metrics_results = {}
                            for metric_name, metric in self.metrics.items():
                                try:
                                    scores = metric.compute(job_ctx)
                                    metrics_results[metric_name] = scores
                                    logger.debug(f"Computed {metric_name} for case {test_case.conv_id}: {scores}")
                                except Exception as e:
                                    logger.error(f"Error computing {metric_name}: {e}")
                                    metrics_results[metric_name] = {}
                            
                            results.append({
                                "conv_id": test_case.conv_id,
                                "model": model_config.name,
                                "prompt": prompt_config.id,
                                "metrics": metrics_results,
                                "response_data": response_data
                            })
                            
                            logger.debug(f"Completed evaluation for case {test_case.conv_id}")
                            
                        except Exception as e:
                            logger.error(f"Error processing case {test_case.conv_id} with {model_config.name}: {e}")
                        
                        pbar.update(1)
        
        # Save complete dataset with responses
        logger.info("Saving complete dataset with model responses")
        self._save_dataset_csv(completed_test_cases)
        
        # Save results
        logger.info("Saving evaluation metrics and costs")
        self._save_metrics_csv(results)
        self._save_latency_cost_csv(results)
        
        logger.info(f"Evaluation complete. Results saved to: {self.output_dir}")
        logger.info(f"Total conversations processed: {len(completed_test_cases)}")
        return results
    
    def _update_test_case_with_response(self, test_case: TestCase, response_data: Dict[str, Any]) -> TestCase:
        """Update test case with model responses"""
        updated_turns = []
        
        # Copy all original turns
        for turn in test_case.turns:
            updated_turns.append(turn.copy())
        
        # Update assistant responses
        responses = response_data.get("responses", [])
        if responses:
            # Find the assistant turn and update it
            for turn in updated_turns:
                if turn["role"] == "assistant" and turn["content"] == "":
                    turn["content"] = responses[0]  # Use the first (and only) response
                    logger.debug(f"Updated assistant response for conv {test_case.conv_id}: {len(responses[0])} chars")
                    break
        
        return TestCase(conv_id=test_case.conv_id, turns=updated_turns)


async def run_evaluation_from_config(config_path: str):
    """Run evaluation from configuration file"""
    framework = EvaluationFramework(config_path)
    return await framework.run_evaluation() 