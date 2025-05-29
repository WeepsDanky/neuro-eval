#!/usr/bin/env python3
"""
Export LoRA fine-tuned model to GGUF format for Ollama deployment.

This script merges LoRA adapters with base model, converts to GGUF format,
and creates necessary files for Ollama deployment.
"""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import commentjson

from weclone.utils.log import logger


def check_and_install_llama_cpp():
    """
    Check if llama.cpp tools are available, provide installation instructions if not.
    """
    # 检查各种可能的转换脚本名称
    convert_scripts = [
        "convert-hf-to-gguf.py", 
        "convert_hf_to_gguf.py",
        "convert-hf-to-gguf",
        "convert_hf_to_gguf"
    ]
    
    convert_script = None
    for script_name in convert_scripts:
        script_path = shutil.which(script_name)
        if script_path:
            convert_script = script_path
            break
    
    # 检查各种可能的量化工具名称
    quantize_tools = [
        "quantize", 
        "llama-quantize",
        "llamacpp-quantize"
    ]
    
    quantize_tool = None
    for tool_name in quantize_tools:
        tool_path = shutil.which(tool_name)
        if tool_path:
            quantize_tool = tool_path
            break
    
    if not convert_script:
        logger.error("未找到 llama.cpp 转换工具")
        logger.info("请按照以下步骤安装 llama.cpp:")
        logger.info("=" * 60)
        logger.info("方法1: 使用便捷安装脚本 (推荐)")
        logger.info("   ./scripts/install_llama_cpp.sh")
        logger.info("")
        logger.info("方法2: 手动下载预编译版本")
        logger.info("   1. 访问: https://github.com/ggerganov/llama.cpp/releases")
        logger.info("   2. 下载适合您系统的预编译版本")
        logger.info("   3. 解压并将工具添加到 PATH")
        logger.info("")
        logger.info("方法3: 从源码编译")
        logger.info("   git clone https://github.com/ggerganov/llama.cpp.git")
        logger.info("   cd llama.cpp && make")
        logger.info("   sudo ln -s $(pwd)/convert-hf-to-gguf.py /usr/local/bin/")
        logger.info("   sudo ln -s $(pwd)/quantize /usr/local/bin/")
        logger.info("=" * 60)
        return False
    
    if not quantize_tool:
        logger.warning("未找到量化工具，将跳过模型量化步骤")
        logger.info("如需量化功能，请确保 llama.cpp 的量化工具在 PATH 中")
    
    logger.info(f"找到转换工具: {convert_script}")
    if quantize_tool:
        logger.info(f"找到量化工具: {quantize_tool}")
    
    return True


def merge_lora_to_base_model(
    base_model_path: str,
    lora_adapter_path: str, 
    output_path: str,
    model_name: str = "qwen"
) -> None:
    """
    Merge LoRA adapter with base model using LLaMA Factory.
    """
    logger.info(f"正在合并 LoRA adapter ({lora_adapter_path}) 到基础模型 ({base_model_path})")
    
    # Check if paths exist
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
    
    if not Path(lora_adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter 路径不存在: {lora_adapter_path}")
    
    # Use LLaMA Factory's export functionality
    try:
        from llamafactory.train.tuner import export_model
        
        # Prepare arguments for model export
        export_args = [
            "--model_name_or_path", base_model_path,
            "--adapter_name_or_path", lora_adapter_path,
            "--template", model_name,
            "--finetuning_type", "lora",
            "--export_dir", output_path,
            "--export_size", "1",
            "--export_device", "cpu",
            "--trust_remote_code", "True"
        ]
        
        # Export merged model
        original_argv = os.sys.argv.copy()
        try:
            os.sys.argv = ["export_model.py"] + export_args
            export_model()
            logger.info(f"模型合并完成，输出路径: {output_path}")
        finally:
            os.sys.argv = original_argv
            
    except ImportError as e:
        logger.error("导入 LLaMA Factory 失败，请确保已正确安装")
        logger.error(f"错误详情: {e}")
        raise
    except Exception as e:
        logger.error(f"模型合并失败: {e}")
        raise


def convert_to_gguf(
    merged_model_path: str,
    output_gguf_path: str,
    quantization: str = "q4_k_m"
) -> None:
    """
    Convert merged model to GGUF format using llama.cpp tools.
    """
    logger.info(f"正在将模型转换为 GGUF 格式...")
    
    if not Path(merged_model_path).exists():
        raise FileNotFoundError(f"合并后的模型路径不存在: {merged_model_path}")
    
    try:
        # 查找转换脚本
        convert_scripts = [
            "convert-hf-to-gguf.py", 
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf",
            "convert_hf_to_gguf"
        ]
        
        convert_script = None
        for script_name in convert_scripts:
            script_path = shutil.which(script_name)
            if script_path:
                convert_script = script_path
                break
        
        if not convert_script:
            raise FileNotFoundError("未找到 llama.cpp 转换工具")
        
        # Create output directory
        Path(output_gguf_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to GGUF (f32 format first)
        temp_gguf_path = output_gguf_path.replace(".gguf", "_f32.gguf")
        
        # 获取绝对路径
        abs_merged_path = str(Path(merged_model_path).absolute())
        abs_gguf_path = str(Path(temp_gguf_path).absolute())
        
        # 构建转换命令
        if convert_script.endswith('.py'):
            convert_cmd = [
                "python", convert_script,
                abs_merged_path,
                "--outfile", abs_gguf_path,
                "--outtype", "f32"
            ]
        else:
            # 如果是二进制文件
            convert_cmd = [
                convert_script,
                abs_merged_path,
                "--outfile", abs_gguf_path,
                "--outtype", "f32"
            ]
        
        logger.info(f"执行转换命令: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True, cwd=Path(output_gguf_path).parent)
        
        if result.returncode != 0:
            logger.error(f"GGUF 转换失败:")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
        
        logger.info("GGUF 基础转换完成")
        
        # Quantize the model if tools available
        quantize_tools = ["quantize", "llama-quantize", "llamacpp-quantize"]
        quantize_tool = None
        for tool_name in quantize_tools:
            tool_path = shutil.which(tool_name)
            if tool_path:
                quantize_tool = tool_path
                break
        
        if quantize_tool and quantization != "f32":
            logger.info(f"正在量化模型: {quantization}")
            quantize_cmd = [
                quantize_tool,
                temp_gguf_path,
                output_gguf_path,
                quantization
            ]
            
            result = subprocess.run(quantize_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"量化失败，使用原始 f32 模型:")
                logger.warning(f"stderr: {result.stderr}")
                shutil.move(temp_gguf_path, output_gguf_path)
            else:
                # Remove the temporary f32 file
                if Path(temp_gguf_path).exists():
                    os.remove(temp_gguf_path)
                logger.info(f"量化完成: {output_gguf_path}")
        else:
            # No quantization, just rename the f32 file
            shutil.move(temp_gguf_path, output_gguf_path)
            if quantize_tool is None:
                logger.info("未找到量化工具，保存为 f32 格式")
            else:
                logger.info(f"GGUF 转换完成 (f32): {output_gguf_path}")
            
    except Exception as e:
        logger.error(f"GGUF 转换失败: {e}")
        raise


def create_ollama_modelfile(
    gguf_path: str,
    output_dir: str,
    model_name: str = "weclone-qwen",
    system_prompt: Optional[str] = None,
    temperature: float = 0.5,
    top_p: float = 0.65,
    repeat_penalty: float = 1.2
) -> str:
    """
    Create Ollama Modelfile for the GGUF model.
    """
    modelfile_path = Path(output_dir) / "Modelfile"
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = "请你扮演一名人类，不要说自己是人工智能"
    
    modelfile_content = f"""FROM {Path(gguf_path).name}

# Set system prompt
SYSTEM \"\"\"{system_prompt}\"\"\"

# Set model parameters
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER repeat_penalty {repeat_penalty}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Template for Qwen model
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}{{{{ .Response }}}}<|im_end|>
\"\"\"
"""
    
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    logger.info(f"Ollama Modelfile 已创建: {modelfile_path}")
    return str(modelfile_path)


def create_deployment_script(
    output_dir: str,
    model_name: str = "weclone-qwen",
    gguf_filename: str = "model.gguf"
) -> str:
    """
    Create deployment script for Windows Ollama.
    """
    script_path = Path(output_dir) / "deploy_to_ollama.bat"
    
    script_content = f"""@echo off
echo 正在部署 WeClone 模型到 Ollama...

REM 检查 Ollama 是否安装
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Ollama，请先安装 Ollama
    echo 下载地址: https://ollama.ai/download/windows
    pause
    exit /b 1
)

REM 创建模型
echo 正在创建 Ollama 模型: {model_name}
ollama create {model_name} -f Modelfile

if %errorlevel% equ 0 (
    echo 模型创建成功！
    echo.
    echo 使用方法:
    echo   ollama run {model_name}
    echo.
    echo 或者在其他应用中使用 API:
    echo   curl http://localhost:11434/api/generate -d "{{\"model\":\"{model_name}\",\"prompt\":\"你好\"}}"
) else (
    echo 模型创建失败，请检查文件是否完整
)

pause
"""
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"部署脚本已创建: {script_path}")
    return str(script_path)


def create_readme(output_dir: str, model_name: str = "weclone-qwen") -> str:
    """
    Create README file with deployment instructions.
    """
    readme_path = Path(output_dir) / "README_DEPLOYMENT.md"
    
    readme_content = f"""# WeClone Ollama 部署说明

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

**方法一：使用部署脚本**
双击运行 `deploy_to_ollama.bat`

**方法二：手动部署**
```cmd
cd /path/to/this/folder
ollama create {model_name} -f Modelfile
```

### 3. 使用模型

**命令行聊天：**
```cmd
ollama run {model_name}
```

**API 调用：**
```bash
curl http://localhost:11434/api/generate \\
  -d '{{"model":"{model_name}","prompt":"你好，请介绍一下自己"}}'
```

**Python 调用示例：**
```python
import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={{"model": "{model_name}", "prompt": "你好"}})
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
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"部署说明已创建: {readme_path}")
    return str(readme_path)


def main():
    """
    Main function to export LoRA model to GGUF format for Ollama.
    """
    logger.info("开始导出 WeClone 模型到 GGUF 格式...")
    
    # Check dependencies first
    if not check_and_install_llama_cpp():
        logger.error("缺少必要的依赖，无法继续导出")
        logger.info("请按照上述说明安装 llama.cpp 工具后再试")
        return
    
    # Load configuration
    config_path = "./settings.jsonc"
    with open(config_path, "r", encoding="utf-8") as f:
        all_config = commentjson.load(f)
    
    common_args = all_config.get("common_args", {})
    infer_args = all_config.get("infer_args", {})
    
    # Get paths from config
    base_model_path = common_args.get("model_name_or_path", "./Qwen2.5-7B-Instruct")
    lora_adapter_path = common_args.get("adapter_name_or_path", "./model_output")
    template = common_args.get("template", "qwen")
    system_prompt = common_args.get("default_system", "请你扮演一名人类，不要说自己是人工智能")
    
    # Get inference parameters
    temperature = infer_args.get("temperature", 0.5)
    top_p = infer_args.get("top_p", 0.65)
    repeat_penalty = infer_args.get("repetition_penalty", 1.2)
    
    # Validate paths
    logger.info(f"基础模型路径: {base_model_path}")
    logger.info(f"LoRA adapter 路径: {lora_adapter_path}")
    
    # Create output directory
    output_dir = Path("./ollama_export")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"输出目录: {output_dir.absolute()}")
    
    try:
        # Step 1: Merge LoRA with base model
        merged_model_dir = output_dir / "merged_model"
        merge_lora_to_base_model(
            base_model_path=base_model_path,
            lora_adapter_path=lora_adapter_path,
            output_path=str(merged_model_dir),
            model_name=template
        )
        
        # Step 2: Convert to GGUF
        gguf_path = output_dir / "model.gguf"
        convert_to_gguf(
            merged_model_path=str(merged_model_dir),
            output_gguf_path=str(gguf_path),
            quantization="q4_k_m"  # Good balance of size and quality
        )
        
        # Step 3: Create Ollama Modelfile
        modelfile_path = create_ollama_modelfile(
            gguf_path=str(gguf_path),
            output_dir=str(output_dir),
            model_name="weclone-qwen",
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )
        
        # Step 4: Create deployment script
        script_path = create_deployment_script(
            output_dir=str(output_dir),
            model_name="weclone-qwen"
        )
        
        # Step 5: Create README
        readme_path = create_readme(
            output_dir=str(output_dir),
            model_name="weclone-qwen"
        )
        
        # Clean up merged model directory (optional, to save space)
        if merged_model_dir.exists():
            shutil.rmtree(merged_model_dir)
            logger.info("已清理临时合并模型文件")
        
        logger.info("=" * 60)
        logger.info("🎉 GGUF 导出完成！")
        logger.info("=" * 60)
        logger.info(f"📁 输出目录: {output_dir.absolute()}")
        logger.info(f"📦 GGUF 模型: {gguf_path}")
        logger.info(f"⚙️  Modelfile: {modelfile_path}")
        logger.info(f"🚀 部署脚本: {script_path}")
        logger.info(f"📖 说明文档: {readme_path}")
        logger.info("")
        logger.info("下一步:")
        logger.info("1. 将整个 ollama_export 文件夹复制到 Windows 系统")
        logger.info("2. 在 Windows 上运行 deploy_to_ollama.bat")
        logger.info("3. 使用 'ollama run weclone-qwen' 开始聊天")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        raise


if __name__ == "__main__":
    main() 