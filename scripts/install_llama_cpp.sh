#!/bin/bash
# WeClone - llama.cpp 安装脚本
# 从源码使用CMake编译安装 Python 脚本和二进制工具

set -e

echo "正在安装 llama.cpp 工具用于 WeClone GGUF 导出功能..."

# 检查系统架构
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Linux" ]]; then
    echo "错误: 此脚本仅支持 Linux 系统"
    echo "请手动安装 llama.cpp 工具"
    exit 1
fi

# 检查必要的工具
if ! command -v git &> /dev/null; then
    echo "错误: 需要 git 来克隆仓库"
    echo "Ubuntu/Debian: sudo apt update && sudo apt install git"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "错误: 需要 cmake 来编译"
    echo "Ubuntu/Debian: sudo apt update && sudo apt install cmake build-essential"
    exit 1
fi

echo "正在从源码安装 llama.cpp 工具..."

# 创建临时目录
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "步骤 1/4: 克隆 llama.cpp 源码..."
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

echo "步骤 2/4: 配置构建环境..."
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "步骤 3/4: 编译量化工具..."
cmake --build . --config Release --target llama-quantize

echo "步骤 4/4: 安装文件..."

# 检查转换脚本是否存在
cd "$TEMP_DIR/llama.cpp"
CONVERT_SCRIPT=""
if [[ -f "convert_hf_to_gguf.py" ]]; then
    CONVERT_SCRIPT="convert_hf_to_gguf.py"
elif [[ -f "convert-hf-to-gguf.py" ]]; then
    CONVERT_SCRIPT="convert-hf-to-gguf.py"
else
    echo "错误: 未找到转换脚本"
    exit 1
fi

echo "找到转换脚本: $CONVERT_SCRIPT"

# 检查编译的量化工具
QUANTIZE_BIN="$TEMP_DIR/llama.cpp/build/bin/llama-quantize"
if [[ ! -f "$QUANTIZE_BIN" ]]; then
    echo "错误: 量化工具编译失败"
    echo "请检查编译输出"
    exit 1
fi

# 安装文件
echo "正在安装转换脚本和量化工具..."
sudo cp "$CONVERT_SCRIPT" /usr/local/bin/convert-hf-to-gguf.py
sudo cp "$QUANTIZE_BIN" /usr/local/bin/quantize
sudo chmod +x /usr/local/bin/convert-hf-to-gguf.py
sudo chmod +x /usr/local/bin/quantize

# 安装 Python 依赖
echo "正在安装 Python 依赖..."
# 跳过requirements.txt，直接安装核心依赖以避免版本冲突
echo "安装核心依赖（跳过requirements.txt以避免版本冲突）..."
if command -v uv &> /dev/null; then
    uv pip install numpy torch transformers sentencepiece protobuf
else
    pip install numpy torch transformers sentencepiece protobuf
fi

# 验证安装
echo "验证安装..."
VERIFY_SUCCESS=true

if command -v convert-hf-to-gguf.py &> /dev/null; then
    echo "✅ 转换脚本安装成功: $(which convert-hf-to-gguf.py)"
else
    echo "❌ 转换脚本安装失败"
    VERIFY_SUCCESS=false
fi

if command -v quantize &> /dev/null; then
    echo "✅ 量化工具安装成功: $(which quantize)"
else
    echo "❌ 量化工具安装失败"
    VERIFY_SUCCESS=false
fi

if [[ "$VERIFY_SUCCESS" == true ]]; then
    echo "✅ llama.cpp 工具安装成功!"
    echo "转换脚本: $(which convert-hf-to-gguf.py)"
    echo "量化工具: $(which quantize)"
    echo "现在您可以使用 'weclone-cli export-to-gguf' 命令了"
else
    echo "❌ 安装失败，请检查上述输出"
    exit 1
fi

# 清理临时文件
cd /
rm -rf "$TEMP_DIR"

echo "安装完成!"
echo ""
echo "如需卸载，请删除以下文件:"
echo "  sudo rm /usr/local/bin/convert-hf-to-gguf.py"
echo "  sudo rm /usr/local/bin/quantize" 