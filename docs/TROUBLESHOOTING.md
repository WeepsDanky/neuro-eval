# WeClone 故障排除指南

## 🔧 GGUF 导出相关问题

### Q: 运行 `weclone-cli export-to-gguf` 提示"未找到 llama.cpp 转换工具"

**A: 安装 llama.cpp 工具**

推荐使用预编译版本安装脚本：
```bash
./scripts/install_llama_cpp.sh
```

如果安装脚本失败，请检查：
1. 网络连接是否正常
2. 是否有 `wget`/`curl` 和 `unzip` 工具
3. 是否有 sudo 权限

手动安装方法：
```bash
# 下载预编译版本
wget https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-*-bin-ubuntu-x64.zip
unzip llama-*-bin-ubuntu-x64.zip
cd llama-*-bin-ubuntu-x64/
sudo ln -s $(pwd)/convert-hf-to-gguf.py /usr/local/bin/
sudo ln -s $(pwd)/quantize /usr/local/bin/
```

### Q: 安装脚本报错"未找到解压后的二进制目录"

**A: 检查下载的文件**

1. 检查下载的文件是否完整：
```bash
ls -la /tmp/tmp.*/llama-bin.zip
```

2. 手动解压查看内容：
```bash
cd /tmp && unzip -l llama-bin.zip
```

3. 如果文件损坏，请重新运行安装脚本

### Q: 转换过程中出现 "GGUF conversion failed"

**A: 检查模型文件和依赖**

1. 确认模型文件存在：
```bash
ls -la ./Qwen2.5-7B-Instruct/
ls -la ./model_output/
```

2. 检查 Python 依赖：
```bash
uv pip install numpy torch transformers sentencepiece protobuf
```

3. 检查磁盘空间（需要约 10-20GB）：
```bash
df -h .
```

### Q: 量化失败，但转换成功

**A: 这是正常的**

如果只是量化失败，模型仍然可以使用，只是文件会稍大一些。量化失败的常见原因：
- 内存不足
- 量化工具版本不兼容

解决方法：
1. 释放更多内存
2. 使用不同的量化格式
3. 跳过量化步骤（使用 f32 格式）

## 💻 系统兼容性问题

### Q: 提示"此脚本仅支持 x86_64 架构"

**A: 使用源码编译**

对于 ARM 或其他架构：
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
sudo ln -s $(pwd)/convert-hf-to-gguf.py /usr/local/bin/
sudo ln -s $(pwd)/quantize /usr/local/bin/
```

### Q: WSL 环境下出现权限问题

**A: 检查文件系统挂载**

1. 确保在 Linux 文件系统中操作：
```bash
pwd  # 应该在 /home/xxx 而不是 /mnt/c/
```

2. 如果必须在 Windows 文件系统中工作，设置权限：
```bash
chmod +x scripts/install_llama_cpp.sh
```

## 🐛 Windows Ollama 部署问题

### Q: Windows 上双击 `deploy_to_ollama.bat` 无反应

**A: 检查 Ollama 安装**

1. 确认 Ollama 已安装：
```cmd
ollama --version
```

2. 手动运行命令：
```cmd
cd C:\path\to\ollama_export
ollama create weclone-qwen -f Modelfile
```

3. 检查文件路径是否包含中文或特殊字符

### Q: 提示"模型创建失败"

**A: 检查文件完整性**

1. 确认 `model.gguf` 文件存在且完整：
```cmd
dir model.gguf
```

2. 检查可用磁盘空间（至少 8GB）

3. 尝试使用管理员权限运行

### Q: 模型运行很慢

**A: 优化性能**

1. 确保有足够内存（推荐 16GB+）
2. 关闭其他占用内存的程序
3. 考虑使用更小的量化格式
4. 在 Modelfile 中调整参数：
```
PARAMETER num_ctx 1024
PARAMETER num_predict 50
```

## 📝 配置问题

### Q: `settings.jsonc` 配置错误

**A: 检查配置格式**

1. 确认 JSON 格式正确：
```bash
python -c "import commentjson; commentjson.load(open('settings.jsonc'))"
```

2. 检查路径是否存在：
```bash
ls -la ./Qwen2.5-7B-Instruct/
ls -la ./model_output/
```

### Q: 找不到 LoRA adapter

**A: 检查训练输出**

1. 确认训练已完成：
```bash
ls -la ./model_output/
```

2. 应该包含以下文件：
- `adapter_config.json`
- `adapter_model.safetensors`
- `README.md`

## 🆘 获取帮助

如果以上方法都无法解决问题：

1. **查看详细日志**：
```bash
weclone-cli export-to-gguf 2>&1 | tee export.log
```

2. **检查系统信息**：
```bash
uname -a
python --version
which python
df -h
free -h
```

3. **提交 Issue**：
访问 [WeClone GitHub Issues](https://github.com/xming521/WeClone/issues) 并提供：
- 错误信息完整输出
- 系统信息
- 配置文件（隐藏敏感信息）

4. **社区支持**：
- 查看已有的 Issues 和讨论
- 阅读项目文档和 README 