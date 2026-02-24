@echo off
title 修复 DocCaptioner 本地环境
cd /d %~dp0

echo ===================================================
echo 正在修复本地环境依赖...
echo 目标: 降级 PyTorch 到 2.4.1 (稳定版) 以解决 nms 算子缺失问题
echo ===================================================

rem 检查 venv 是否存在
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到 venv 环境，请先运行 install.bat
    pause
    exit /b
)

echo [1/3] 卸载当前 PyTorch 版本...
venv\Scripts\pip uninstall -y torch torchvision torchaudio

echo [2/3] 安装 PyTorch 2.4.1 (CUDA 12.4)...
venv\Scripts\pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

echo [3/3] 验证环境...
venv\Scripts\python -c "import torch; import torchvision; print(f'Torch: {torch.__version__}'); print(f'TorchVision: {torchvision.__version__}'); print('CUDA:', torch.cuda.is_available()); try: import torchvision.ops; print('NMS Operator check: OK'); except Exception as e: print(f'NMS Check Failed: {e}')"

echo.
echo ===================================================
echo 修复完成！如果 NMS Check 显示 OK，请重新运行 run.bat
echo ===================================================
pause