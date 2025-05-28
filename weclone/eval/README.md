# WeClone Evaluation Framework

A comprehensive evaluation system for testing conversational AI models with multi-metric benchmarking, plugin architecture, and extensive data persistence.

## Features

- **Multi-Metric Benchmarking**: Interaction Fluency, Sentiment Satisfaction, Task Success Rate, Latency, and Cost analysis
- **Plugin Architecture**: Extensible metric system for custom evaluation criteria
- **Multi-Model Testing**: Compare different models and prompt variants simultaneously
- **Comprehensive Data Persistence**: All inputs, outputs, metrics, and metadata stored in structured formats
- **Configuration-Driven**: YAML/JSON configuration files for easy setup and modification
- **Async Processing**: Efficient parallel evaluation execution

## Quick Start

### 1. Install Dependencies

The framework requires `pyyaml` for configuration parsing:

```bash
# Using uv (recommended)
uv add pyyaml

# Or using pip
pip install pyyaml
```

### 2. Create Configuration File

Create an evaluation configuration file (e.g., `my_eval.yaml`):

```yaml
batch_name: "my_evaluation"

prompts:
  - id: "friendly_assistant"
    version: "v1.0"
    content: "你是一个友善、有用的AI助手。"

models:
  - name: "weclone:local"
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 150
    host: "http://127.0.0.1:8005/v1"
    api_key: "sk-test"

cases:
  - file: "dataset/test_data.json"

metrics:
  - interaction_fluency
  - sentiment_satisfaction
  - task_success
  - latency
  - cost
```

### 3. Run Evaluation

```bash
# Start the API server (in one terminal)
python -m weclone.cli server

# Run evaluation (in another terminal)
python -m weclone.cli eval-framework --config my_eval.yaml
```

## Configuration Reference

### Model Configuration

```yaml
models:
  - name: "model_identifier"
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 150
      # Any other OpenAI API parameters
    host: "http://localhost:8005/v1"  # API endpoint
    api_key: "your-api-key"           # Optional API key
```

### Prompt Configuration

```yaml
prompts:
  - id: "prompt_identifier"
    version: "v1.0"
    content: "Your system prompt content here"
  
  # Or load from file
  - id: "prompt_from_file"
    version: "v1.1"
    file: "path/to/prompt.txt"
```

### Test Cases

#### JSON Format (Compatible with existing test_data.json)
```json
{
  "questions": [
    ["你好", "今天天气怎么样？"],
    ["帮我写个故事"]
  ]
}
```

#### JSONL Format (New conversational format)
```jsonl
{"conv_id": "greeting", "conversation": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "今天过得怎么样？"}, {"role": "assistant", "content": ""}]}
```

### Available Metrics

1. **Interaction Fluency (互动流畅度)**
   - `interrupt_count`: Number of user interruptions
   - `timeout_resend_count`: Number of timeout-triggered resends
   - `avg_turn_interval`: Average time between conversation turns

2. **Sentiment Satisfaction (情感满意度)**
   - `post_chat_rating`: Overall satisfaction score (1-5)
   - `sentiment_score`: Sentiment analysis score

3. **Task Success (任务成功率)**
   - `retrieval_precision`: Information retrieval accuracy
   - `gen_bleu`: Text generation quality (BLEU-like score)
   - `function_call_accuracy`: Function/tool usage accuracy

4. **Latency (延迟)**
   - `first_token_ms`: Time to first token (milliseconds)
   - `full_response_ms`: Total response time (milliseconds)

5. **Cost (成本)**
   - `prompt_tokens`: Input token count
   - `completion_tokens`: Output token count
   - `usd_cost`: Estimated cost in USD

## Output Structure

Results are saved in the `eval_runs/` directory:

```
eval_runs/
└── 20241228T142030Z_abc12345/
    ├── run_meta.json          # Run metadata and configuration
    ├── dataset.csv            # Test cases in CSV format
    ├── metrics.csv            # Detailed metric results
    └── latency_cost.csv       # Performance and cost data
```

### Output Files

#### run_meta.json
Contains run metadata, configuration, and enabled metrics.

#### dataset.csv
```csv
conv_id,turn_idx,role,content
0,0,user,"你好"
0,1,assistant,"你好！有什么可以帮您的吗？"
```

#### metrics.csv
```csv
run_id,conv_id,model,prompt,metric,value
20241228T142030Z_abc12345,0,weclone:local,friendly_assistant,sentiment_satisfaction.post_chat_rating,4.2
```

#### latency_cost.csv
```csv
run_id,conv_id,model,n_tokens_prompt,n_tokens_completion,latency_ms,cost_usd
20241228T142030Z_abc12345,0,weclone:local,45,23,1250,0.000234
```

## Custom Metrics

Create custom metrics by extending the `BaseMetric` class:

```python
from weclone.eval.framework import BaseMetric, JobContext

class CustomMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "custom_metric"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text"]
    
    def compute(self, job_ctx: JobContext) -> Dict[str, Any]:
        # Your custom metric logic here
        return {"custom_score": 0.85}
```

## Integration with Existing CLI

The evaluation framework integrates seamlessly with the existing WeClone CLI:

```bash
# List all available commands
python -m weclone.cli --help

# Run the comprehensive evaluation
python -m weclone.cli eval-framework --config eval_config.yaml

# Run the simple test (existing functionality)
python -m weclone.cli test-model
```

## Best Practices

1. **Start the API Server**: Always ensure the API server is running before evaluation
2. **Use Version Control**: Track configuration file versions for reproducible results
3. **Monitor Costs**: Set `max_usd` limits in configuration to control API costs
4. **Parallel Processing**: Adjust `parallel` setting based on API rate limits
5. **Data Backup**: Archive evaluation results for historical comparison

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure API server is running on the correct port
2. **Missing Dependencies**: Install `pyyaml` for configuration parsing
3. **File Not Found**: Check that test case files exist and paths are correct
4. **API Rate Limits**: Reduce `parallel` setting or add delays

### Debug Mode

Enable detailed logging by setting the log level:

```python
import logging
logging.getLogger("weclone.eval").setLevel(logging.DEBUG)
```

## Examples

See the `weclone/eval/config/` directory for example configurations:

- `example_eval.yaml`: Comprehensive example with all features
- `additional_cases.jsonl`: Example JSONL test cases

## Architecture

The evaluation framework follows a plugin architecture:

```
EvaluationFramework
├── Configuration Loader (YAML/JSON)
├── Test Case Loader (JSON/JSONL)
├── Model Manager (OpenAI compatible)
├── Metric Plugins
│   ├── InteractionFluencyMetric
│   ├── SentimentSatisfactionMetric
│   ├── TaskSuccessMetric
│   ├── LatencyMetric
│   └── CostMetric
└── Data Persistence (CSV/JSON)
```

This design allows for easy extension and modification of evaluation criteria without changing core functionality. 