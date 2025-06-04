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
    
    def load_cases_from_files(self) -> List[TestCase]:
        """Load test cases from configuration files with automatic format detection"""
        test_cases = []
        cases_config = self.config.get("cases", [])
        
        for case_config in cases_config:
            if isinstance(case_config, dict) and "file" in case_config:
                file_path = Path(case_config["file"])
                
                # Load based on file extension
                if file_path.suffix.lower() == ".json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    questions = data.get("questions", [])
                    for i, question_set in enumerate(questions):
                        turns = []
                        # Add all user inputs first
                        for question in question_set:
                            turns.append({"role": "user", "content": question})
                        # Add one assistant response placeholder at the end
                        turns.append({"role": "assistant", "content": ""})
                        test_cases.append(TestCase(conv_id=str(i), turns=turns))
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Support debug mode with limited test cases
        debug_limit = self.config.get("debug", {}).get("max_cases", None)
        if debug_limit and debug_limit > 0:
            test_cases = test_cases[:debug_limit]
            logger.info(f"Debug mode: Limited to {len(test_cases)} test cases")
        
        return test_cases
    
    def save_all_results(self, results: List[Dict[str, Any]], completed_test_cases: List[TestCase]):
        """把所有输入和输出数据保存在一个 csv 中, 包括所有指标和 judge 输出， 一行对应一个 case/model/prompt 组合"""
        # Save run metadata
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
        
        # Save metadata as JSON
        metadata_path = self.output_dir / "run_meta.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Prepare unified results table
        unified_rows = []
        
        # Create a mapping from conv_id to completed test case for easy lookup
        case_map = {case.conv_id: case for case in completed_test_cases}
        
        # prepare data for saving to csv
        for result in results:
            conv_id = result["conv_id"]
            test_case = case_map.get(conv_id)
            
            # Basic info
            row = {
                "run_id": self.run_id,
                "conv_id": conv_id,
                "model": result["model"],
                "prompt": result["prompt"],
            }
            
            # Input: combine all user messages
            user_messages = []
            assistant_message = ""
            if test_case:
                for turn in test_case.turns:
                    if turn["role"] == "user":
                        user_messages.append(turn["content"])
                    elif turn["role"] == "assistant" and turn["content"]:
                        assistant_message = turn["content"]
            
            row["input"] = " | ".join(user_messages)  # Join multiple user inputs
            row["output"] = assistant_message
            
            # Response metadata
            response_data = result.get("response_data", {})
            token_usage = response_data.get("token_usage", {})
            row["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
            row["completion_tokens"] = token_usage.get("completion_tokens", 0)
            row["latency_ms"] = response_data.get("total_time_ms", 0)
            
            # Add all benchmark scores dynamically
            benchmark_results = result.get("benchmark_results", {})
            for benchmark_name, benchmark_result in benchmark_results.items():
                if hasattr(benchmark_result, 'metrics'):
                    for metric_name, metric_value in benchmark_result.metrics.items():
                        # Create column name like "chathumanscore_nat_score"
                        column_name = f"{benchmark_name}_{metric_name}"
                        row[column_name] = metric_value
            
            # Add GPT judge outputs to the row
            gpt_judge_outputs = result.get("gpt_judge_outputs", [])
            if gpt_judge_outputs:
                # Convert judge data to JSON string for storage in CSV
                judge_data = gpt_judge_outputs[0].get("judge_data", {})
                row["gpt_judge_output"] = json.dumps(judge_data, ensure_ascii=False)
            else:
                row["gpt_judge_output"] = ""
            
            unified_rows.append(row)
        
        # Convert to list format for CSV saving
        if unified_rows:
            # Get all unique column names
            all_columns = set()
            for row in unified_rows:
                all_columns.update(row.keys())
            
            # Sort columns for consistent output
            fixed_columns = ["run_id", "conv_id", "model", "prompt", "input", "output", 
                           "prompt_tokens", "completion_tokens", "latency_ms"]
            benchmark_columns = sorted([col for col in all_columns if col not in fixed_columns and col != "gpt_judge_output"])
            header = fixed_columns + benchmark_columns + ["gpt_judge_output"]
            
            # Convert rows to list format
            csv_rows = []
            for row in unified_rows:
                csv_row = [row.get(col, "") for col in header]
                csv_rows.append(csv_row)
            
            # Save unified results as CSV
            results_path = self.output_dir / "results.csv"
            with open(results_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(csv_rows)
            
            logger.info(f"Saved {len(unified_rows)} results with {len(header)} columns")
            logger.info(f"Columns: {', '.join(header)}")
        else:
            logger.info("No results to save")
    
    async def _run_conversation(self, model_config: ModelConfig, prompt_config: PromptConfig, test_case: TestCase) -> Dict[str, Any]:
        """Run a single conversation with timing and token tracking, 返回 response_data 和 gpt_judge_outputs"""
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
    
    async def run_evaluation(self):
        """Run the complete evaluation, 返回 results 和 completed_test_cases"""
        logger.info(f"Starting evaluation run: {self.run_id}")
        
        # Load test cases
        test_cases = self.load_cases_from_files()
        logger.info(f"Loaded {len(test_cases)} test cases")
        
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
                            gpt_judge_outputs = []  # 收集 GPT judge 输出用于保存
                            for benchmark_name, benchmark in self.benchmarks.items():
                                try:
                                    result = benchmark.compute(job_ctx)
                                    benchmark_results[benchmark_name] = result
                                    
                                    # 如果是 ChatHumanScore benchmark 且有 GPT judge 输出，收集它
                                    if (benchmark_name == 'chathumanscore' and 
                                        hasattr(result, 'metadata') and 
                                        result.metadata and 
                                        'raw_judge_output' in result.metadata):
                                        gpt_judge_output = result.metadata['raw_judge_output']
                                        gpt_judge_outputs.append({
                                            'conv_id': test_case.conv_id,
                                            'model': model_config.name,
                                            'prompt': prompt_config.id,
                                            'judge_data': gpt_judge_output
                                        })
                                    
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
                                "response_data": response_data,
                                "gpt_judge_outputs": gpt_judge_outputs  # 添加 GPT judge 输出
                            })
                            
                            logger.debug(f"Completed evaluation for case {test_case.conv_id}")
                            
                        except Exception as e:
                            logger.error(f"Error processing case {test_case.conv_id} with {model_config.name}: {e}")
                        
                        pbar.update(1)
        
        # Save all results using the unified method
        logger.info("Saving all results")
        self.save_all_results(results, completed_test_cases)
        
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