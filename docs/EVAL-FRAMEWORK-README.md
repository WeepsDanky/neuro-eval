# WeClone Evaluation Framework

A robust, extensible evaluation system for testing conversational AI models with comprehensive benchmarking capabilities.

## Features

- **Modular Benchmark Architecture**: Each benchmark is a separate, configurable module
- **Multi-Metric Evaluation**: Interaction fluency, sentiment satisfaction, task success, latency, and cost metrics
- **YAML/JSON Configuration**: Flexible configuration management
- **Comprehensive Data Persistence**: Structured CSV output with timestamped runs
- **Multi-Model Support**: Test multiple models and prompt variants simultaneously
- **Debug Mode**: Limited test cases for rapid iteration

## Architecture

### Benchmark System

The evaluation framework uses a plugin-based benchmark system where each benchmark is implemented as a separate module:

```
weclone/eval/
├── benchmark/
│   ├── __init__.py              # Benchmark registry
│   ├── base.py                  # Base classes and interfaces
│   ├── interaction_fluency.py   # Conversation flow metrics
│   ├── sentiment_satisfaction.py # User satisfaction analysis
│   ├── task_success.py          # Task completion accuracy
│   ├── latency.py              # Response timing metrics
│   └── cost.py                 # API usage cost analysis
├── framework.py                 # Core evaluation engine
└── config/                     # Configuration examples
    ├── debug_test.yaml
    └── enhanced_benchmark_test.yaml
```

### Available Benchmarks

1. **Interaction Fluency** (`interaction_fluency`)
   - Measures conversation flow and interruptions
   - Metrics: interrupt_count, timeout_resend_count, avg_turn_interval

2. **Sentiment Satisfaction** (`sentiment_satisfaction`) 
   - Analyzes user satisfaction through sentiment analysis
   - Metrics: post_chat_rating (1-5), sentiment_score (-1 to 1)

3. **Task Success** (`task_success`)
   - Evaluates task completion accuracy
   - Metrics: retrieval_precision, gen_bleu, function_call_accuracy

4. **Latency** (`latency`)
   - Measures response timing performance
   - Metrics: first_token_ms, full_response_ms, throughput_tokens_per_sec

5. **Cost** (`cost`)
   - Calculates API usage costs and efficiency
   - Metrics: prompt_tokens, completion_tokens, usd_cost, cost_per_token

## Configuration

### Basic Configuration

```yaml
models:
  - name: "weclone:local"
    params:
      model: "gpt-3.5-turbo"
      max_tokens: 150
      temperature: 0.7

prompts:
  - id: "default_system"
    version: "1.0"
    content: "你是一个友好的AI助手。"

cases:
  - file: "dataset/test_data.json"

metrics:
  - "latency"
  - "cost"
```

### Advanced Benchmark Configuration

Each benchmark can be customized with specific parameters:

```yaml
benchmark_configs:
  interaction_fluency:
    interrupt_threshold_ms: 300
    timeout_threshold_ms: 25000
  
  sentiment_satisfaction:
    positive_words: ["好", "棒", "满意", "喜欢"]
    negative_words: ["差", "糟糕", "不满", "讨厌"]
    base_rating: 3.5
    rating_sensitivity: 0.8
  
  cost:
    model_pricing:
      "gpt-3.5-turbo":
        prompt: 0.0005
        completion: 0.0015
    excellent_cost_per_token: 0.000005
```

## Usage

### Command Line Interface

```bash
# Run evaluation with specific config
uv run python -m weclone.cli eval --config weclone/eval/config/debug_test.yaml

# Debug mode with limited test cases
uv run python -m weclone.cli eval --config weclone/eval/config/debug_test.yaml --debug 3
```

### Programmatic Usage

```python
from weclone.eval.framework import run_evaluation_from_config

# Run evaluation
results = await run_evaluation_from_config("config/my_evaluation.yaml")

# Access benchmark results
for result in results:
    conv_id = result["conv_id"]
    latency_metrics = result["benchmark_results"]["latency"].metrics
    cost_metrics = result["benchmark_results"]["cost"].metrics
    
    print(f"Conversation {conv_id}:")
    print(f"  Latency: {latency_metrics['full_response_ms']}ms")
    print(f"  Cost: ${cost_metrics['usd_cost']}")
```

## Creating Custom Benchmarks

To create a new benchmark, inherit from `BaseBenchmark`:

```python
from weclone.eval.benchmark.base import BaseBenchmark, BenchmarkResult
from weclone.eval.framework import JobContext

class MyCustomBenchmark(BaseBenchmark):
    @property
    def _get_name(self) -> str:
        return "my_custom_benchmark"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text"]
    
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        # Your custom logic here
        metrics = {"my_metric": 0.85}
        
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics
        )
```

Then register it in `benchmark/__init__.py`:

```python
AVAILABLE_BENCHMARKS = {
    # ... existing benchmarks
    'my_custom_benchmark': MyCustomBenchmark
}
```

## Output Format

The framework generates structured outputs in timestamped directories:

```
eval_runs/
└── 20241201T143022Z_a1b2c3d4/
    ├── run_meta.json           # Run metadata and configuration
    ├── dataset.csv             # Complete conversation dataset
    ├── benchmark_results.csv   # All benchmark metrics
    └── latency_cost.csv       # Latency and cost summary
```

### CSV Output Structure

**benchmark_results.csv**:
```csv
run_id,conv_id,model,prompt,benchmark,metric,value
20241201T143022Z_a1b2c3d4,0,weclone:local,default_system,latency,full_response_ms,1234.56
```

**dataset.csv**:
```csv
conv_id,turn_idx,role,content
0,0,user,你好
0,1,assistant,你好！有什么可以帮助你的吗？
```

## Requirements

```bash
uv add openai pyyaml tqdm
```

## Debug Mode

For rapid iteration during development:

```yaml
debug:
  max_cases: 3  # Limit to 3 test cases
```

This allows quick testing of configuration changes without running the full evaluation suite.

## API Server Setup

Before running evaluations, ensure the API server is running:

```bash
# Start the WeClone API server
uv run python -m weclone.cli server

# Server runs on http://127.0.0.1:8005/v1 by default
```

## Configuration Schema

Each benchmark provides its own configuration schema accessible via `get_config_schema()`. This enables IDE autocomplete and validation for benchmark-specific settings.

The modular architecture makes it easy to:
- Add new benchmarks without modifying core framework code
- Configure benchmarks independently
- Maintain and test benchmarks in isolation
- Share benchmark implementations across projects 