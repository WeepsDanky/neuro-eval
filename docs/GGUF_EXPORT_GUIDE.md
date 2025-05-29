# WeClone GGUF 导出指南

本指南介绍如何将 WeClone 训练的 LoRA 模型导出为 GGUF 格式，并在 Windows Ollama 中部署。

## 🎯 概述

GGUF 是一种高效的模型格式，特别适合在本地设备上运行大型语言模型。通过导出为 GGUF 格式，您可以：

- 在 Windows 上使用 Ollama 运行您的数字分身
- 减少模型文件大小（通过量化）
- 获得更快的推理速度
- 在没有 Python/CUDA 环境的机器上运行

## 📋 前置要求

### WSL (Linux) 环境
- 已完成 WeClone 模型训练
- 安装了 `build-essential` 和 `cmake`
- 有足够的磁盘空间（约 10-20GB）

### Windows 环境
- 安装了 [Ollama](https://ollama.ai/download/windows)
- 至少 8GB RAM（推荐 16GB+）

## 🔧 安装依赖

### 方法1: 使用便捷脚本（推荐）

新的安装脚本会自动下载 llama.cpp 的预编译二进制文件，避免了编译过程中的各种问题：

```bash
# 在 WeClone 项目根目录下运行
./scripts/install_llama_cpp.sh
```

这个脚本会：
- 自动检测系统架构
- 下载最新的预编译二进制文件
- 创建符号链接到系统路径
- 安装必要的 Python 依赖

### 方法2: 手动下载预编译版本

```bash
# 1. 访问 GitHub releases 页面
# https://github.com/ggerganov/llama.cpp/releases

# 2. 下载适合您系统的预编译版本
# 例如：llama-b4037-bin-ubuntu-x64.zip

# 3. 解压并创建符号链接
unzip llama-*-bin-ubuntu-x64.zip
cd llama-*-bin-ubuntu-x64/
sudo ln -s $(pwd)/convert-hf-to-gguf.py /usr/local/bin/
sudo ln -s $(pwd)/quantize /usr/local/bin/

# 4. 安装 Python 依赖
uv pip install numpy torch transformers sentencepiece protobuf
```

### 方法3: 从源码编译（不推荐）

仅在预编译版本不可用时使用：

```bash
# 1. 安装构建工具（如果未安装）
sudo apt update
sudo apt install build-essential cmake

# 2. 克隆和编译 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# 3. 安装 Python 依赖
uv pip install -r requirements.txt

# 4. 创建符号链接
sudo ln -s $(pwd)/convert-hf-to-gguf.py /usr/local/bin/
sudo ln -s $(pwd)/quantize /usr/local/bin/
```

## 💡 为什么使用预编译版本？

- ⚡ **更快安装**: 避免编译过程，几分钟内完成安装
- 🛡️ **更可靠**: 官方测试的二进制文件，减少兼容性问题
- 🔧 **更简单**: 无需安装编译器和构建工具
- 💾 **更小体积**: 只下载必要的二进制文件

## 🚀 导出流程

### 1. 验证模型文件

确保以下文件存在：
```
./Qwen2.5-7B-Instruct/    # 基础模型
./model_output/           # LoRA adapter
./settings.jsonc          # 配置文件
```

### 2. 执行导出命令

```bash
weclone-cli export-to-gguf
```

导出过程包括：
1. **模型合并**: 将 LoRA adapter 合并到基础模型
2. **格式转换**: 转换为 GGUF 格式
3. **模型量化**: 压缩模型文件（q4_k_m 格式）
4. **生成部署文件**: 创建 Modelfile 和部署脚本

### 3. 检查输出

导出完成后，您会在 `ollama_export/` 目录下看到：

```
ollama_export/
├── model.gguf                 # GGUF 模型文件
├── Modelfile                  # Ollama 配置文件
├── deploy_to_ollama.bat       # Windows 部署脚本
└── README_DEPLOYMENT.md       # 部署说明
```

## 💾 传输到Windows

### 方法1: 网络共享
```bash
# 如果配置了网络共享
cp -r ollama_export /mnt/c/Users/YourName/Desktop/
```

### 方法2: 压缩传输
```bash
# 压缩文件夹
tar -czf ollama_export.tar.gz ollama_export/

# 传输到 Windows 系统
# 在 Windows 上解压并使用
```

## 🎮 在Windows中部署

### 1. 确保 Ollama 已安装

从 https://ollama.ai/download/windows 下载并安装 Ollama

### 2. 部署模型

**方法1: 使用脚本（推荐）**
双击运行 `deploy_to_ollama.bat`

**方法2: 手动部署**
```cmd
# 打开命令提示符，进入文件夹
cd C:\path\to\ollama_export

# 创建 Ollama 模型
ollama create weclone-qwen -f Modelfile
```

### 3. 验证部署

```cmd
# 测试模型
ollama run weclone-qwen
```

## 💬 使用模型

### 命令行聊天
```cmd
ollama run weclone-qwen
```

### API 调用
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"weclone-qwen","prompt":"你好，请介绍一下自己"}'
```

### Python 集成
```python
import requests

def chat_with_weclone(message):
    response = requests.post('http://localhost:11434/api/generate', 
        json={"model": "weclone-qwen", "prompt": message})
    return response.json()

# 使用示例
result = chat_with_weclone("你好，今天天气怎么样？")
print(result)
```

## 🔧 参数调整

编辑 `Modelfile` 中的参数：

```dockerfile
# 控制输出随机性
PARAMETER temperature 0.7

# 核采样参数
PARAMETER top_p 0.9

# 重复惩罚
PARAMETER repeat_penalty 1.1
```

修改后重新运行部署命令：
```cmd
ollama create weclone-qwen -f Modelfile
```

## 🐛 故障排除

### 常见问题

**Q: 转换失败，提示找不到 convert-hf-to-gguf.py**
A: 运行依赖安装脚本：`./scripts/install_llama_cpp.sh`

**Q: Windows 上显示"模型创建失败"**
A: 检查文件路径是否正确，确保 model.gguf 文件存在

**Q: 内存不足错误**
A: 7B 模型需要至少 8GB RAM，可以尝试更小的模型或增加虚拟内存

**Q: 模型响应很慢**
A: 这是正常的，CPU 推理比 GPU 慢。可以考虑使用更小的量化格式

### 性能优化

1. **使用 GPU 加速**（如果支持）：
   ```cmd
   # 在 Windows 上安装 CUDA 版本的 Ollama
   ```

2. **调整并发参数**：
   ```dockerfile
   PARAMETER num_ctx 2048    # 减少上下文长度
   PARAMETER num_predict 100 # 限制输出长度
   ```

3. **选择更激进的量化**：
   修改导出脚本中的 `quantization="q4_0"` 获得更小的文件

## 📚 参考资源

- [Ollama 官方文档](https://github.com/ollama/ollama)
- [GGUF 格式说明](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md)
- [llama.cpp 项目](https://github.com/ggerganov/llama.cpp)
- [WeClone 项目](https://github.com/xming521/WeClone) 