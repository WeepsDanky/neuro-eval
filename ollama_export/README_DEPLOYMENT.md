# WeClone Ollama 部署说明

## 文件说明

- `model.gguf`: 转换后的 GGUF 格式模型文件
- `Modelfile`: Ollama 模型配置文件
- `deploy_to_ollama.bat`: Windows 部署脚本
- `README_DEPLOYMENT.md`: 本说明文件

## 部署步骤

### 1. 安装 Ollama (Windows)

如果还未安装 Ollama，请从官网下载安装：
https://ollama.ai/download/windows

### 2. 部署模型

将整个文件夹复制到 Windows 系统中，然后：

**从 WSL 复制到 Windows：**
```bash
# 复制到 Windows 用户桌面
cp -r ollama_export /mnt/c/Users/$(whoami)/Desktop/

# 或复制到指定位置
cp -r ollama_export /mnt/d/Projects/weclone
```

**方法一：使用部署脚本**
双击运行 `deploy_to_ollama.bat`

**方法二：手动部署**
```cmd
cd /path/to/this/folder
ollama create weclone-qwen -f Modelfile
```

### 3. 使用模型

**命令行聊天：**
```cmd
ollama run weclone-qwen
```

**API 调用：**
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"weclone-qwen","prompt":"你好，请介绍一下自己"}'
```

**Python 调用示例：**
```python
import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={"model": "weclone-qwen", "prompt": "你好"})
print(response.json())
```

## 参数调整

如需调整模型参数，可以编辑 `Modelfile` 文件中的 PARAMETER 部分：

- `temperature`: 控制输出随机性 (0.1-2.0)
- `top_p`: 核采样参数 (0.1-1.0) 
- `repeat_penalty`: 重复惩罚 (1.0-1.5)

修改后重新运行部署命令即可。

## 故障排除

1. **模型文件过大**: GGUF 文件可能很大，确保有足够磁盘空间
2. **权限问题**: 在 Windows 上可能需要管理员权限运行
3. **内存不足**: 7B 模型大约需要 4-8GB RAM

## 技术支持

如有问题，请访问 WeClone 项目页面：
https://github.com/xming521/WeClone
