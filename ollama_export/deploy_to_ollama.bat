@echo off
echo Deploying WeClone Model to Ollama...

REM 检查 Ollama 是否安装
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Ollama is not found, please install Ollama first
    echo Download URL: https://ollama.ai/download/windows
    pause
    exit /b 1
)

REM Create Model
echo Creating Ollama model: weclone-qwen
ollama create weclone-qwen -f Modelfile

if %errorlevel% equ 0 (
    echo Model created successful! 
    echo.
    echo How to use:
    echo   ollama run weclone-qwen
    echo.
    echo Or use API in other softwares:
    echo   curl http://localhost:11434/api/generate -d "{"model":"weclone-qwen","prompt":"你好"}"
) else (
    echo Failed to create model, please check is the file corrupted...
)

pause
