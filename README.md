<!-- This project is forked from WeClone (https://github.com/xming521/WeClone) -->
<!-- 本项目从 WeClone (https://github.com/xming521/WeClone) fork 而来 -->

# WeClone 使用指南

## 🚀 快速开始

WeClone 提供了完整的命令行界面来处理从数据准备到模型部署的整个流程。所有命令都通过 `weclone-cli` 调用。

### 基本语法
```bash
weclone-cli [COMMAND] [OPTIONS]
```

## 📝 可用命令列表

### 1. `make-dataset` - 数据集生成
**功能**: 处理聊天记录CSV文件，生成问答对数据集
```bash
weclone-cli make-dataset
```
- 读取 `./dataset/csv/` 目录下的聊天记录文件
- 生成训练用的问答对数据集
- 支持隐私信息过滤和数据清洗
- 配置参数在 `settings.jsonc` 的 `make_dataset_args` 中设置

### 2. `train-sft` - 模型微调
**功能**: 使用准备好的数据集对模型进行监督微调
```bash
weclone-cli train-sft
```
- 基于LLaMA Factory进行LoRA微调
- 支持单卡/多卡训练
- 训练参数在 `settings.jsonc` 的 `train_sft_args` 中配置

### 3. `export-to-gguf` - 模型导出
**功能**: 将LoRA微调模型导出为GGUF格式，用于Ollama部署
```bash
weclone-cli export-to-gguf
```
- 自动合并LoRA adapter和基础模型
- 转换为GGUF格式并支持量化
- 生成Windows Ollama部署包
- 包含Modelfile和部署脚本
- **依赖**: 需要安装 llama.cpp 工具
- **详细指南**: 请参考 [GGUF导出指南](docs/GGUF_EXPORT_GUIDE.md)

### 4. `webchat-demo` - Web界面测试
**功能**: 启动 Web UI 与微调后的模型进行交互测试
```bash
weclone-cli webchat-demo
```
- 提供友好的Web界面进行对话测试
- 可调整 temperature、top_p 等推理参数
- 用于验证微调效果

### 5. `server` - API服务
**功能**: 启动API服务，提供模型推理接口
```bash
weclone-cli server
```
- 启动OpenAI兼容的API服务
- 默认监听 `http://127.0.0.1:8005/v1`
- 支持聊天机器人集成

### 6. `test-model` - 模型测试
**功能**: 使用常见聊天问题测试模型性能
```bash
weclone-cli test-model
```
- 使用预定义的测试问题集评估模型
- 生成测试结果报告 `test_result-my.txt`
- **注意**: 需要先启动 `server` 命令

### 7. `eval-framework` - 综合评估框架
**功能**: 运行多维度、多指标的全面模型评估
```bash
weclone-cli eval-framework --config <配置文件路径>
```

**参数说明**:
- `--config, -c`: 评估配置文件路径（必需）
  - 支持 YAML 和 JSON 格式
  - 示例: `weclone/eval/config/simple_test.yaml`

**评估指标**:
- **互动流畅度**: 中断次数、超时重发、平均轮次间隔
- **情感满意度**: 聊天后评分、情感分数
- **任务成功率**: 检索精度、生成质量、函数调用准确性
- **延迟性能**: 首token时间、完整响应时间、吞吐量
- **成本分析**: token使用量、USD成本、成本效率

**输出结果**:
- 保存到 `eval_runs/<时间戳>/` 目录
- 包含详细的CSV数据和运行元数据
- 显示平均指标摘要

### 8. `eval-model` - 验证集评估 
**功能**: 使用从训练数据中划分出来的验证集进行评估
```bash
weclone-cli eval-model
```
- 用于模型训练过程中的性能监控
- 基于验证集数据评估模型效果

## 🔬 评估框架架构

WeClone 提供了功能强大的模块化评估框架，支持对任何 OpenAI 兼容的模型进行全面的多维度评估。

### 框架特性

- **🔧 模块化基准系统**: 每个基准都是独立的可配置模块
- **📊 多指标评估**: 交互流畅度、情感满意度、任务成功率、延迟性能、成本分析
- **⚙️ 灵活配置**: 支持 YAML/JSON 配置文件
- **💾 数据持久化**: 结构化 CSV 输出，时间戳标记的运行记录
- **🌐 多模型支持**: 同时测试多个模型和提示变体
- **🐛 调试模式**: 限制测试案例数量，快速迭代开发

### 支持的模型类型

评估框架支持任何 OpenAI 兼容的 API，包括：

- **本地模型**: WeClone 微调后的模型 (`http://127.0.0.1:8005/v1`)
- **OpenAI 模型**: GPT-3.5, GPT-4 等 (`https://api.openai.com/v1`)
- **第三方 API**: OpenRouter、Anthropic 代理等
  - DeepSeek: `https://openrouter.ai/api/v1`
  - 通义千问: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - 智谱 GLM: `https://open.bigmodel.cn/api/paas/v4`
- **自部署模型**: vLLM、FastChat、Ollama 等兼容服务

### 评估基准详情

#### 1. 交互流畅度 (`interaction_fluency`)
- **中断计数**: 对话中的中断次数
- **超时重发**: 响应超时导致的重发次数  
- **平均轮次间隔**: 用户-助手轮次之间的平均时间间隔

#### 2. 情感满意度 (`sentiment_satisfaction`)
- **聊天后评分**: 1-5 分的主观满意度评分
- **情感分数**: -1 到 1 的情感极性分析

#### 3. 任务成功率 (`task_success`)
- **检索精度**: 信息检索的准确性
- **生成质量**: BLEU 分数等生成质量指标
- **函数调用准确性**: 工具使用的正确率

#### 4. 延迟性能 (`latency`)
- **首 Token 时间**: 首个 Token 生成延迟
- **完整响应时间**: 完成整个响应的时间
- **吞吐量**: 每秒 Token 生成数量

#### 5. 成本分析 (`cost`)
- **Token 使用量**: 输入和输出 Token 统计
- **USD 成本**: 基于模型定价的成本计算
- **成本效率**: 每 Token 成本分析

### 配置示例

#### 基础配置
```yaml
batch_name: "model_comparison_test"

# 提示配置
prompts:
  - id: "default_system"
    version: "v1.0" 
    content: "你是一个友善、有用的AI助手。"

# 模型配置 - 支持多种 API
models:
  # 本地微调模型
  - name: "weclone:local"
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 150
    host: "http://127.0.0.1:8005/v1"
    api_key: "sk-test"
    
  # OpenRouter DeepSeek
  - name: "deepseek:chat-v3"
    params:
      model: "deepseek/deepseek-chat-v3-0324"
      temperature: 0.7
      max_tokens: 150
    host: "https://openrouter.ai/api/v1"
    api_key: "${OPENROUTER_API_KEY}"
    
  # OpenAI 官方
  - name: "openai:gpt-4"
    params:
      model: "gpt-4"
      temperature: 0.7
      max_tokens: 150
    host: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"

# 测试数据
cases:
  - file: "dataset/test_data.json"

# 启用的评估指标
metrics:
  - interaction_fluency
  - sentiment_satisfaction  
  - task_success
  - latency
  - cost

# 执行设置
parallel: 2
max_usd: 10.0
timeout_seconds: 60
```

#### 高级基准配置
```yaml
# 基准模块自定义配置
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
      "gpt-4":
        prompt: 0.03
        completion: 0.06
      "deepseek/deepseek-chat-v3-0324":
        prompt: 0.0014
        completion: 0.0028
    excellent_cost_per_token: 0.000005
```

### 使用方式

#### 命令行评估
```bash
# 运行评估
weclone-cli eval-framework --config weclone/eval/config/deepseek_openrouter_test.yaml

# 调试模式（限制测试案例）
weclone-cli eval-framework --config weclone/eval/config/debug_test.yaml
```

#### 环境变量配置
```bash
# 设置 API 密钥
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENAI_API_KEY="your_openai_key"
export DASHSCOPE_API_KEY="your_qwen_key"
```

## 📋 配置格式详细说明

### 配置文件结构

评估配置文件支持 **YAML** 和 **JSON** 格式，主要包含以下部分：

```yaml
# 基本信息
batch_name: "string"           # 评估批次名称 (必填)

# 核心配置部分
prompts: []                    # 提示词配置列表 (必填)
models: []                     # 模型配置列表 (必填)  
cases: []                      # 测试数据配置列表 (必填)
metrics: []                    # 评估指标列表 (必填)

# 可选配置部分
benchmark_configs: {}          # 基准模块自定义配置 (可选)
debug: {}                      # 调试模式配置 (可选)
parallel: int                  # 并发数 (可选, 默认: 1)
max_usd: float                 # 最大成本限制 (可选, 默认: 无限制)
timeout_seconds: int           # 请求超时时间 (可选, 默认: 30)
output_formats: []             # 输出格式列表 (可选)
```

### 核心配置参数

#### 1. `prompts` - 提示词配置
定义系统提示词和对话设置：

```yaml
prompts:
  - id: "string"              # 提示词唯一标识符 (必填)
    version: "string"         # 版本号 (必填)
    content: "string"         # 提示词内容 (content 和 file 二选一)
    file: "path/to/file"      # 提示词文件路径 (content 和 file 二选一)
```

**示例**：
```yaml
prompts:
  - id: "default_system"
    version: "v1.0"
    content: "你是一个友善、有用的AI助手。"
  
  - id: "casual_chat"
    version: "v1.1" 
    file: "prompts/casual_system.txt"
```

#### 2. `models` - 模型配置
定义要评估的模型和参数：

```yaml
models:
  - name: "string"            # 模型名称标识 (必填)
    host: "string"            # API 基础 URL (可选, 默认: http://127.0.0.1:8005/v1)
    api_key: "string"         # API 密钥 (可选, 支持环境变量 ${VAR_NAME})
    params:                   # 模型参数 (必填)
      model: "string"         # 模型标识符 (必填)
      temperature: float      # 采样温度 (可选, 0.0-2.0)
      max_tokens: int         # 最大生成token数 (可选)
      top_p: float           # 核采样参数 (可选, 0.0-1.0)
      top_k: int             # Top-K采样 (可选)
      frequency_penalty: float # 频率惩罚 (可选, -2.0-2.0)
      presence_penalty: float  # 存在惩罚 (可选, -2.0-2.0)
      stop: [string]          # 停止词列表 (可选)
```

**支持的 API 类型**：
- **本地 WeClone**: `http://127.0.0.1:8005/v1`
- **OpenAI**: `https://api.openai.com/v1`
- **OpenRouter**: `https://openrouter.ai/api/v1`
- **阿里通义**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **智谱 GLM**: `https://open.bigmodel.cn/api/paas/v4`

**示例**：
```yaml
models:
  # 本地微调模型
  - name: "weclone-local"
    host: "http://127.0.0.1:8005/v1"
    api_key: "sk-test"
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 512
      top_p: 0.9
  
  # OpenRouter 第三方模型
  - name: "deepseek-v3"
    host: "https://openrouter.ai/api/v1"
    api_key: "${OPENROUTER_API_KEY}"
    params:
      model: "deepseek/deepseek-chat-v3-0324"
      temperature: 0.5
      max_tokens: 1024
```

#### 3. `cases` - 测试数据配置
定义测试数据来源：

```yaml
cases:
  - file: "path/to/data.json"   # JSON 格式数据文件
  - file: "path/to/data.jsonl"  # JSONL 格式数据文件
```

**支持的数据格式**：

**JSON 格式** (兼容现有 test_data.json):
```json
{
  "questions": [
    ["问题1", "问题2"],
    ["另一组问题"]
  ]
}
```

**JSONL 格式** (每行一个对话):
```jsonl
{"conversation": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": ""}]}
{"conversation": [{"role": "user", "content": "今天天气怎么样？"}, {"role": "assistant", "content": ""}]}
```

#### 4. `metrics` - 评估指标
指定要使用的评估基准：

```yaml
metrics:
  - "interaction_fluency"     # 交互流畅度
  - "sentiment_satisfaction"  # 情感满意度
  - "task_success"           # 任务成功率
  - "latency"                # 延迟性能
  - "cost"                   # 成本分析
  - "chathumanscore"         # 人类化评分 (需额外依赖)
```

### 高级配置参数

#### 5. `benchmark_configs` - 基准自定义配置
为每个基准提供专门的配置：

```yaml
benchmark_configs:
  # 交互流畅度配置
  interaction_fluency:
    interrupt_threshold_ms: 300        # 中断阈值 (毫秒)
    timeout_threshold_ms: 25000       # 超时阈值 (毫秒)
  
  # 情感满意度配置  
  sentiment_satisfaction:
    positive_words: ["好", "棒", "满意"]  # 积极词汇
    negative_words: ["差", "糟糕", "不满"] # 消极词汇
    base_rating: 3.5                    # 基础评分
    rating_sensitivity: 0.8             # 评分敏感度
  
  # 任务成功率配置
  task_success:
    bleu_weight: 0.4                   # BLEU 权重
    precision_weight: 0.3              # 精确率权重
    recall_weight: 0.3                 # 召回率权重
  
  # 延迟性能配置
  latency:
    first_token_excellent_ms: 500      # 优秀首token时间
    first_token_good_ms: 1000         # 良好首token时间
    full_response_excellent_ms: 2000   # 优秀完整响应时间
    full_response_good_ms: 5000       # 良好完整响应时间
  
  # 成本分析配置
  cost:
    model_pricing:                     # 自定义模型定价
      "gpt-4":
        prompt: 0.03                   # 输入token价格 (USD/1K)
        completion: 0.06               # 输出token价格 (USD/1K)
      "deepseek/deepseek-chat-v3-0324":
        prompt: 0.0014
        completion: 0.0028
    excellent_cost_per_token: 0.000005 # 优秀成本阈值
    good_cost_per_token: 0.00005      # 良好成本阈值
    
  # ChatHumanScore 配置 (需安装额外依赖)
  chathumanscore:
    enable_grammar_check: true         # 启用语法检查
    enable_semantic_analysis: true     # 启用语义分析
    enable_gpt_judge: false           # 启用GPT评判
    max_grammar_error_rate: 0.05      # 最大语法错误率
    max_repeat_ratio: 0.30            # 最大重复率
    human_review_threshold: 5.0       # 人工审核阈值
    score_weights:                    # 评分权重
      naturalness: 0.25              # 自然度
      affective_alignment: 0.20       # 情感对齐
      diversity: 0.15                # 多样性
      context_cohesion: 0.20         # 上下文粘性
      human_signal: 0.20             # 人类信号
```

#### 6. `debug` - 调试模式配置
快速迭代开发的调试选项：

```yaml
debug:
  max_cases: 3                      # 限制测试案例数量
  verbose: true                     # 详细日志输出
  save_intermediate: true           # 保存中间结果
```

#### 7. 执行控制参数

```yaml
# 并发控制
parallel: 2                         # 并发workers数量 (默认: 1)

# 成本控制
max_usd: 10.0                      # 最大成本限制 USD (默认: 无限制)

# 超时控制
timeout_seconds: 60                # 单个请求超时时间 (默认: 30秒)

# 输出控制
output_formats:                    # 输出格式 (默认: ["csv"])
  - "csv"                         # CSV 格式
  - "json"                        # JSON 格式
  - "summary_report"              # 摘要报告
```

### 环境变量支持

配置文件支持环境变量替换，使用 `${VAR_NAME}` 语法：

```yaml
models:
  - name: "openai-gpt4"
    api_key: "${OPENAI_API_KEY}"     # 从环境变量读取
    host: "${OPENAI_BASE_URL}"       # 可选的自定义端点
```

**常用环境变量**：
```bash
# 设置环境变量
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"  
export DASHSCOPE_API_KEY="your_qwen_key"
export GLM_API_KEY="your_glm_key"
```

### 完整配置示例

```yaml
batch_name: "comprehensive_model_evaluation"

prompts:
  - id: "system_default"
    version: "v1.0"
    content: "你是一个友善、专业的AI助手。"
  
  - id: "casual_chat"
    version: "v1.1"
    content: "你是用户的朋友，用轻松的语气聊天。"

models:
  - name: "weclone-local"
    host: "http://127.0.0.1:8005/v1"
    api_key: "sk-test"
    params:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 512
  
  - name: "deepseek-v3"
    host: "https://openrouter.ai/api/v1"
    api_key: "${OPENROUTER_API_KEY}"
    params:
      model: "deepseek/deepseek-chat-v3-0324"
      temperature: 0.5
      max_tokens: 1024
      top_p: 0.9

cases:
  - file: "dataset/test_data.json"
  - file: "dataset/additional_cases.jsonl"

metrics:
  - "interaction_fluency"
  - "sentiment_satisfaction"
  - "latency"
  - "cost"

benchmark_configs:
  cost:
    model_pricing:
      "deepseek/deepseek-chat-v3-0324":
        prompt: 0.0014
        completion: 0.0028
    excellent_cost_per_token: 0.000005

debug:
  max_cases: 5

parallel: 2
max_usd: 5.0
timeout_seconds: 60
output_formats: ["csv", "json"]
```

### 输出结果

评估结果保存在 `eval_runs/<时间戳>/` 目录：

```
eval_runs/20241201T143022Z_a1b2c3d4/
├── run_meta.json              # 运行元数据和配置
├── dataset.csv                # 完整对话数据集
├── benchmark_results.csv      # 所有基准指标
└── latency_cost.csv          # 延迟和成本汇总
```

#### CSV 输出格式

**benchmark_results.csv**:
```csv
run_id,conv_id,model,prompt,benchmark,metric,value
20241201T143022Z_a1b2c3d4,0,deepseek:chat-v3,default_system,latency,full_response_ms,1234.56
20241201T143022Z_a1b2c3d4,0,deepseek:chat-v3,default_system,cost,usd_cost,0.0028
```

**latency_cost.csv**:
```csv
run_id,conv_id,model,n_tokens_prompt,n_tokens_completion,latency_ms,cost_usd
20241201T143022Z_a1b2c3d4,0,deepseek:chat-v3,45,128,1234.56,0.0028
```

### 自定义基准

创建自定义评估基准：

```python
from weclone.eval.benchmark.base import BaseBenchmark, BenchmarkResult
from weclone.eval.framework import JobContext

class CustomBenchmark(BaseBenchmark):
    @property 
    def name(self) -> str:
        return "custom_benchmark"
    
    def required_artifacts(self) -> List[str]:
        return ["conversation_text"]
        
    def compute(self, job_ctx: JobContext) -> BenchmarkResult:
        # 自定义评估逻辑
        metrics = {"custom_metric": 0.85}
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics
        )
```

在 `weclone/eval/benchmark/__init__.py` 中注册：
```python
AVAILABLE_BENCHMARKS = {
    # ... 现有基准
    'custom_benchmark': CustomBenchmark
}
```

### 集成工作流

评估框架与 WeClone 其他组件无缝集成：

1. **训练后评估**: 微调完成后自动评估模型性能
2. **A/B 测试**: 比较不同模型、参数配置的效果
3. **持续监控**: 定期评估生产环境模型表现
4. **模型选择**: 基于评估结果选择最佳模型配置

## 🔧 典型工作流程

### 完整的数字分身创建流程:

1. **准备数据**
   ```bash
   # 将聊天记录CSV文件放入 ./dataset/csv/ 目录
   weclone-cli make-dataset
   ```

2. **训练模型**
   ```bash
   weclone-cli train-sft
   ```

3. **测试效果**
   ```bash
   # Web界面测试
   weclone-cli webchat-demo
   
   # 或启动API服务进行测试
   weclone-cli server
   ```

4. **性能评估**
   ```bash
   # 基础测试
   weclone-cli test-model
   
   # 综合评估
   weclone-cli eval-framework --config weclone/eval/config/simple_test.yaml
   ```

5. **导出到Ollama (可选)**
   ```bash
   # 方法1: 使用便捷安装脚本 (推荐) - 下载预编译版本
   ./scripts/install_llama_cpp.sh
   
   # 方法2: 手动下载预编译版本
   # 访问 https://github.com/ggerganov/llama.cpp/releases
   # 下载适合您系统的预编译版本，解压并添加到 PATH
   
   # 方法3: 从源码编译（仅在预编译版本不可用时使用）
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && make
   sudo ln -s $(pwd)/convert-hf-to-gguf.py /usr/local/bin/
   sudo ln -s $(pwd)/quantize /usr/local/bin/
   
   # 导出模型为GGUF格式
   weclone-cli export-to-gguf
   
   # 将 ollama_export 文件夹复制到Windows系统
   # 在Windows上双击运行 deploy_to_ollama.bat
   ```

6. **部署应用**
   ```bash
   # 保持API服务运行，供聊天机器人调用
   weclone-cli server
   ```

## ⚙️ 配置文件

- **主配置**: `settings.jsonc` - 包含所有模块的配置参数
- **评估配置**: `weclone/eval/config/*.yaml` - 评估框架专用配置

## 💡 使用提示

- 所有命令需要启动虚拟环境，然后在项目根目录执行
- 确保 `settings.jsonc` 配置文件存在且配置正确
- 评估框架需要先启动 API 服务
- 使用 `--help` 查看命令详细帮助: `uv run python -m weclone.cli [COMMAND] --help`

---


<p align="center">
  <a href="https://blog.051088.xyz/2025/05/14/WeClone-%E7%94%A8%E5%BE%AE%E4%BF%A1%E8%81%8A%E5%A4%A9%E8%AE%B0%E5%BD%95%E6%89%93%E9%80%A0%E8%87%AA%E5%B7%B1%E7%9A%84AI%E6%95%B0%E5%AD%97%E5%88%86%E8%BA%AB/" target="_blank">
    Windows部署指南
  </a>
  <a>|</a>
  <a href="https://blog.051088.xyz/posts/weclone-linux-tutorial/" target="_blank">
    Linux部署指南【保姆级】
  </a>
</p>

> [!IMPORTANT]
> <h3> WhatsApp and Telegram chat logs integration for digital avatar creation is coming ! </h3>

## ✨核心功能
- 💫 涵盖打造数字分身的全链路方案，包括聊天数据导出、预处理、模型训练、部署
- 💬 使用微信聊天记录微调LLM，让大模型有"那味儿"
- 🔗 绑定到微信、QQ、Telegram、企微、飞书机器人，实现自己的数字分身
- 🛡️ 隐私信息过滤，本地化微调部署，数据安全可控

## 📋特性与说明

> [!IMPORTANT]
> - WeClone仍在快速迭代期，当前效果不代表最终效果。  
> - 微调LLM效果很大程度取决于模型大小、聊天数据的数量和质量，理论上模型越大，数据越多，效果越好。   
> - Windows环境未进行严格测试，可以使用WSL作为运行环境。详细教程可点击[Windows部署指南](https://blog.051088.xyz/2025/05/14/WeClone-%E7%94%A8%E5%BE%AE%E4%BF%A1%E8%81%8A%E5%A4%A9%E8%AE%B0%E5%BD%95%E6%89%93%E9%80%A0%E8%87%AA%E5%B7%B1%E7%9A%84AI%E6%95%B0%E5%AD%97%E5%88%86%E8%BA%AB/)查看。

### 硬件要求

项目默认使用Qwen2.5-7B-Instruct模型，LoRA方法对sft阶段微调，大约需要16GB显存。也可以使用[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E6%A8%A1%E5%9E%8B)支持的其他模型和方法。

需要显存的估算值：
| 方法                             | 精度 |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |  16  |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |


## 环境搭建
1.cuda安装(已安装可跳过，**要求版本12.4及以上**)：[LLaMA Factory](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#cuda) 

2.建议使用 [uv](https://docs.astral.sh/uv/)安装依赖，这是一个非常快速的 Python 环境管理器。安装uv后，您可以使用以下命令创建一个新的Python环境并安装依赖项，注意这不包含音频克隆功能的依赖：
```bash
git clone https://github.com/xming521/WeClone.git
cd WeClone
uv venv .venv --python=3.10
source .venv/bin/activate # windows下执行 .venv\Scripts\activate
uv pip install --group main -e . 
```
> [!TIP]
> 如果要使用最新的模型进行微调，需要手动安装最新版LLaMA Factory：`uv pip install --upgrade git+https://github.com/hiyouga/LLaMA-Factory.git`,同时其他依赖版本也可能需要修改，例如vllm pytorch transforms

3.将配置文件模板复制一份并重命名为`settings.jsonc`，后续配置修改在此文件进行：
```bash
cp settings.template.jsonc settings.jsonc
```
> [!NOTE]
> 训练以及推理相关配置统一在文件`settings.jsonc`

4.使用以下命令测试CUDA环境是否正确配置并可被PyTorch识别，Mac不需要：
```bash
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available());"
```

5.（可选）安装FlashAttention，加速训练和推理：`uv pip install flash-attn --no-build-isolation`

## 模型下载
```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git
```
下载有问题使用其他方式下载：[模型的下载](https://www.modelscope.cn/docs/models/download)


## 数据准备

请使用[PyWxDump](https://github.com/xaoyaoo/PyWxDump)提取微信聊天记录（不支持4.0版本微信）。可以先将手机的聊天记录迁移（备份）到电脑，数据量更多一些。下载软件并解密数据库后，点击聊天备份，导出类型为CSV，可以导出多个联系人（不建议使用群聊记录），然后将导出的位于`wxdump_tmp/export` 的 `csv` 文件夹放在`./dataset`目录即可，也就是不同人聊天记录的文件夹一起放在 `./dataset/csv`。   

## 数据预处理

- 项目默认去除了数据中的手机号、身份证号、邮箱、网址。还在`settings.jsonc`中提供了一个禁用词词库`blocked_words`，可以自行添加需要过滤的词句（会默认去掉包括禁用词的整句）。
> [!IMPORTANT]
> 🚨 请一定注意保护个人隐私，不要泄露个人信息！

- 执行以下命令对数据进行处理，可以根据自己的聊天风格修改settings.jsonc的`make_dataset_args`。
```bash
weclone-cli make-dataset
```
- 目前仅支持时间窗口策略，根据`single_combine_time_window`将单人连续消息通过逗号连接合并为一句，根据`qa_match_time_window`匹配问答对。
- 可以启用`clean_dataset`中的`enable_clean`选项，对数据进行清洗，以达到更好效果。* 当前系统支持使用 `llm judge` 对聊天记录进行打分，提供 **vllm 离线推理** 和 **API 在线推理** 两种方式。可通过将 `settings.jsonc` 文件中的 `"online_llm_clear": false` 修改为 `true` 来启用 API 在线推理模式，并配置相应的 `base_url`、`llm_api_key`、`model_name` 等参数。所有兼容 OpenAI 接口的模型均可接入。
- 在获得 `llm 打分分数分布情况` 后，可通过设置 `accept_score` 参数筛选可接受的分数区间，同时可适当降低 `train_sft_args` 中的 `lora_dropout` 参数，以提升模型的拟合效果。

## 配置参数并微调模型

- (可选)修改 `settings.jsonc` 的 `model_name_or_path` 和 `template` 选择本地下载好的其他模型。  
- 修改`per_device_train_batch_size`以及`gradient_accumulation_steps`来调整显存占用。  
- 可以根据自己数据集的数量和质量修改`train_sft_args`的`num_train_epochs`、`lora_rank`、`lora_dropout`等参数。

### 单卡训练
```bash
weclone-cli train-sft
```
多卡环境单卡训练，需要先执行 `export CUDA_VISIBLE_DEVICES=0`

### 多卡训练
取消`settings.jsonc`中`deepspeed`行代码注释，使用以下命令多卡训练：
```bash
uv pip install deepspeed
deepspeed --num_gpus=使用显卡数量 weclone/train/train_sft.py
```

### 使用浏览器demo简单推理
可以在这一步测试出合适的temperature、top_p值，修改settings.jsonc的`infer_args`后，供后续推理时使用。
```bash
weclone-cli webchat-demo
```

### 使用接口进行推理

```bash
weclone-cli server
```

### 使用常见聊天问题测试
不包含询问个人信息的问题，仅有日常聊天。测试结果在test_result-my.txt。
```bash
weclone-cli server
weclone-cli test-model
```

## 🖼️ 微调效果
使用Qwen2.5-14B-Instruct模型，大概3万条处理后的有效数据，loss降到了3.5左右的效果。
<details>
<summary>截图</summary>
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/0775ec52-452b-485f-9785-c6eb7b277132" alt="alt text" style="width: 48%; min-width: 150px;">
  <img src="https://github.com/user-attachments/assets/8c7628b5-da70-4c37-9e51-fdfb0eadd2df" alt="alt text" style="width: 48%; min-width: 150px;">
  <img src="https://github.com/user-attachments/assets/523aa742-2aa3-40e9-bd67-b98b336e83a8" alt="alt text" style="width: 48%; min-width: 150px;">
  <img src="https://github.com/user-attachments/assets/dabf0603-dcc4-4a47-b5c3-2bbc036820d9" alt="alt text" style="width: 48%; min-width: 150px;">
</div>
</details>


## 🤖 部署到聊天机器人

### AstrBot

[AstrBot](https://github.com/AstrBotDevs/AstrBot) 是易上手的多平台 LLM 聊天机器人及开发框架 ✨ 平台支持 QQ、QQ频道、Telegram、微信、企微、飞书。      

使用步骤：
1. 部署 AstrBot
2. 在 AstrBot 中部署消息平台
3. 执行 `weclone-cli server` 启动api服务
4. 在 AstrBot 中新增服务提供商，类型选择OpenAI，API Base URL 根据AstrBot部署方式填写（例如docker部署可能为http://172.17.0.1:8005/v1） ，模型填写gpt-3.5-turbo,API Key随意填写一个
5. 微调后不支持工具调用，请先关掉默认的工具，消息平台发送指令： `/tool off all`，否则会没有微调后的效果。 
6. 根据微调时使用的default_system，在 AstrBot 中设置系统提示词。
![5](https://github.com/user-attachments/assets/19de7072-076a-4cdf-8ae6-46b9b89f536a)
> [!IMPORTANT]
> 检查api_service的日志，尽量保证大模型服务请求的参数和微调时一致，tool插件能力都关掉。
7. 调整采样参数，例如temperature、top_p、top_k等
[配置自定义的模型参数](https://astrbot.app/config/model-config.html#%E9%85%8D%E7%BD%AE%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0)

### LangBot

[LangBot](https://github.com/RockChinQ/LangBot) 是一个开源的接入全球多种即时通信平台的 LLM 机器人平台，适合各种场景使用。

1. [部署 LangBot](https://github.com/RockChinQ/LangBot#-%E5%BC%80%E5%A7%8B%E4%BD%BF%E7%94%A8)
2. 在 LangBot 中添加一个机器人
4. 在模型页添加新模型，名称`gpt-3.5-turbo`，供应商选择 OpenAI，填写 请求 URL 为 WeClone 的地址，详细连接方式可以参考[文档](https://docs.langbot.app/zh/workshop/network-details.html)，API Key 任意填写。

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/fc167dea-7c93-4d94-9c5f-db709d0320ba" />

6. 在流水线配置中选择刚才添加的模型，或修改提示词配置

<img width="400px" alt="image" src="https://github.com/user-attachments/assets/dbb0fd0a-f760-42db-acd0-bb99c859b52e" />

## 📌 路线图
- [ ] 更丰富的上下文：包括上下文对话、聊天对象信息、时间等 + 思考
- [ ] Memory 支持
- [ ] 支持多模态
- [ ] 数据增强
- [ ] 支持GUI

## 问题解决
- 微调问题：[LLaMA-Factory| FAQs | 常见问题](https://github.com/hiyouga/LLaMA-Factory/issues/4614) 或者更方便的 [![更方便的Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hiyouga/LLaMA-Factory)

## ❤️ 贡献代码

欢迎任何 Issues/Pull Requests！

你可以通过查看Issues或帮助审核 PR（拉取请求）来贡献。对于新功能的添加，请先通过 Issue 讨论。   
运行`uv pip install --group dev -e .`安装开发依赖。   
项目使用`pytest`测试(测试脚本待完善)，`pyright`检查类型，`ruff`检查代码格式。


## ⚠️ 免责声明
> [!CAUTION]
> 请勿用于非法用途，否则后果自负。
<details>
<summary>1. 使用目的</summary>

* 本项目仅供学习交流使用，**请勿用于非法用途**，**请勿用于非法用途**，**请勿用于非法用途**，否则后果自负。
* 用户理解并同意，任何违反法律法规、侵犯他人合法权益的行为，均与本项目及其开发者无关，后果由用户自行承担。

2. 使用期限

* 您应该在下载保存使用本项目的24小时内，删除本项目的源代码和程序；超出此期限的任何使用行为，一概与本项目及其开发者无关。

3. 操作规范

* 本项目仅允许在授权情况下使用数据训练，严禁用于非法目的，否则自行承担所有相关责任；用户如因违反此规定而引发的任何法律责任，将由用户自行承担，与本项目及其开发者无关。
* 严禁用于窃取他人隐私，严禁用于窃取他人隐私，严禁用于窃取他人隐私，否则自行承担所有相关责任。

4. 免责声明接受

* 下载、保存、进一步浏览源代码或者下载安装、编译使用本程序，表示你同意本警告，并承诺遵守它;

5. 禁止用于非法测试或渗透

* 禁止利用本项目的相关技术从事非法测试或渗透，禁止利用本项目的相关代码或相关技术从事任何非法工作，如因此产生的一切不良后果与本项目及其开发者无关。
* 任何因此产生的不良后果，包括但不限于数据泄露、系统瘫痪、侵犯隐私等，均与本项目及其开发者无关，责任由用户自行承担。

6. 免责声明修改

* 本免责声明可能根据项目运行情况和法律法规的变化进行修改和调整。用户应定期查阅本页面以获取最新版本的免责声明，使用本项目时应遵守最新版本的免责声明。

7. 其他

* 除本免责声明规定外，用户在使用本项目过程中应遵守相关的法律法规和道德规范。对于因用户违反相关规定而引发的任何纠纷或损失，本项目及其开发者不承担任何责任。

* 请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。

</details>
请用户慎重阅读并理解本免责声明的所有内容，确保在使用本项目时严格遵守相关规定。

<br>  
<br>  
<br>  

## ⭐ Star History
> [!TIP] 
> 如果本项目对您有帮助，或者您关注本项目的未来发展，请给项目 Star，谢谢 

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=xming521/WeClone&type=Date)](https://www.star-history.com/#xming521/WeClone&Date)

</div>


<div align="center"> 克隆我们，保留灵魂的芬芳 </div>
