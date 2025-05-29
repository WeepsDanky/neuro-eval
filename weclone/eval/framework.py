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
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

try:
    import yaml
except ImportError:
    yaml = None

from openai import OpenAI
from tqdm import tqdm

from weclone.utils.log import logger
from weclone.utils.config import load_config
from .types import ModelConfig, PromptConfig, TestCase, JobContext
from .benchmark import AVAILABLE_BENCHMARKS, BaseBenchmark, BenchmarkResult


class EvaluationFramework:
    """Main evaluation framework"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.benchmarks = self._initialize_benchmarks()
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
    
    def _initialize_benchmarks(self) -> Dict[str, BaseBenchmark]:
        """Initialize benchmark plugins"""
        benchmarks = {}
        enabled_benchmarks = self.config.get("metrics", [])
        
        for benchmark_name in enabled_benchmarks:
            if benchmark_name not in AVAILABLE_BENCHMARKS:
                logger.warning(f"Unknown benchmark: {benchmark_name}")
                continue
            
            # Get benchmark configuration
            benchmark_config = self.config.get("benchmark_configs", {}).get(benchmark_name, {})
            
            # Initialize benchmark
            benchmark_class = AVAILABLE_BENCHMARKS[benchmark_name]
            benchmark = benchmark_class(config=benchmark_config)
            benchmarks[benchmark_name] = benchmark
            
            logger.debug(f"Initialized benchmark: {benchmark_name}")
        
        logger.info(f"Loaded {len(benchmarks)} benchmarks: {list(benchmarks.keys())}")
        return benchmarks
    
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
            "conversation_json": test_case.turns,  # Add conversation_json artifact for ChatHumanScore
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
            "enabled_benchmarks": list(self.benchmarks.keys()),
            "benchmark_descriptions": {
                name: benchmark.get_description() 
                for name, benchmark in self.benchmarks.items()
            }
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
    
    def _save_benchmark_results_csv(self, results: List[Dict[str, Any]]):
        """Save benchmark results in CSV format"""
        with open(self.output_dir / "benchmark_results.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "conv_id", "model", "prompt", "benchmark", "metric", "value"])
            
            for result in results:
                for benchmark_name, benchmark_result in result["benchmark_results"].items():
                    for metric_name, metric_value in benchmark_result.metrics.items():
                        writer.writerow([
                            self.run_id,
                            result["conv_id"],
                            result["model"],
                            result["prompt"],
                            benchmark_name,
                            metric_name,
                            metric_value
                        ])
    
    def _save_latency_cost_csv(self, results: List[Dict[str, Any]]):
        """Save latency and cost data in CSV format"""
        with open(self.output_dir / "latency_cost.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "conv_id", "model", "n_tokens_prompt", "n_tokens_completion", "latency_ms", "cost_usd"])
            
            for result in results:
                latency_result = result["benchmark_results"].get("latency")
                cost_result = result["benchmark_results"].get("cost")
                
                latency_ms = latency_result.metrics.get("full_response_ms", 0) if latency_result else 0
                prompt_tokens = cost_result.metrics.get("prompt_tokens", 0) if cost_result else 0
                completion_tokens = cost_result.metrics.get("completion_tokens", 0) if cost_result else 0
                cost_usd = cost_result.metrics.get("usd_cost", 0) if cost_result else 0
                
                writer.writerow([
                    self.run_id,
                    result["conv_id"],
                    result["model"],
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    cost_usd
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
                            
                            # Compute benchmark results
                            benchmark_results = {}
                            for benchmark_name, benchmark in self.benchmarks.items():
                                try:
                                    result = benchmark.compute(job_ctx)
                                    benchmark_results[benchmark_name] = result
                                    logger.debug(f"Computed {benchmark_name} for case {test_case.conv_id}: {result.metrics}")
                                except Exception as e:
                                    logger.error(f"Error computing {benchmark_name}: {e}")
                                    benchmark_results[benchmark_name] = BenchmarkResult(
                                        benchmark_name=benchmark_name,
                                        metrics={}
                                    )
                            
                            results.append({
                                "conv_id": test_case.conv_id,
                                "model": model_config.name,
                                "prompt": prompt_config.id,
                                "benchmark_results": benchmark_results,
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
        logger.info("Saving benchmark results and costs")
        self._save_benchmark_results_csv(results)
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