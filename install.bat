@echo off
title DocCaptioner v1.0 安装程序
cls

echo ===================================================
echo       DocCaptioner v1.0 - 环境安装程序
echo       作者: Doc_workBox
echo ===================================================
echo.

REM 1. 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python 或未添加到 PATH 环境变量。
    echo 请从 python.org 安装 Python 3.10 或 3.11。
    pause
    exit /b
)

REM 2. 创建虚拟环境 (Create Venv)
if not exist "venv" (
    echo [信息] 正在创建虚拟环境...
    python -m venv venv
) else (
    echo [信息] 虚拟环境已存在。
)

REM 3. 激活虚拟环境 (Activate Venv)
call venv\Scripts\activate

REM 4. 安装核心依赖 (Install Core Requirements)
echo.
echo [信息] 正在安装核心依赖 (NiceGUI, AI Utils)...
pip install -r requirements.txt
echo.

REM 5. GPU 选择菜单 (GPU Selection Menu)
echo ===================================================
echo           选择您的硬件加速类型
echo ===================================================
echo 1. NVIDIA 显卡 (CUDA 12.8) - 推荐 RTX 系列
echo 2. AMD 显卡 (ROCm) - 适用于 RX 6000/7000 系列 (Windows)
echo 3. 仅 CPU (速度较慢，作为备用或纯API模式)
echo.
set /p gpu_choice="请输入数字 (1-3): "

if "%gpu_choice%"=="1" goto install_nvidia
if "%gpu_choice%"=="2" goto install_amd
if "%gpu_choice%"=="3" goto install_cpu
goto end

:install_nvidia
echo.
echo [信息] 正在为 NVIDIA 安装 PyTorch (CUDA 12.8)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
goto end

:install_amd
echo.
echo [信息] 正在为 AMD 安装 PyTorch (ROCm Nightly)...
echo [提示] 使用通用的 gfx1100 安装配置。如果您有特定的显卡型号，
echo        请查看 README 中关于 "TheRock" 的说明。
echo.
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1100-all/
goto end

:install_cpu
echo.
echo [信息] 正在为 CPU 安装 PyTorch...
pip install torch torchvision torchaudio
goto end

:end
echo.
echo ===================================================
echo    安装完成！请运行 start.bat 启动程序。
echo ===================================================
pause