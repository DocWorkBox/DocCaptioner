@echo off
title DocCaptioner v1.1 ��װ����
cls

echo ===================================================
echo       DocCaptioner v1.1 - ������װ����
echo       ����: Doc_workBox
echo ===================================================
echo.

REM 1. ��� Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [����] δ��⵽ Python ��δ���ӵ� PATH ����������
    echo ��� python.org ��װ Python 3.10 �� 3.11��
    pause
    exit /b
)

REM 2. �������⻷�� (Create Venv)
if not exist "venv" (
    echo [��Ϣ] ���ڴ������⻷��...
    python -m venv venv
) else (
    echo [��Ϣ] ���⻷���Ѵ��ڡ�
)

REM 3. �������⻷�� (Activate Venv)
call venv\Scripts\activate

REM 4. ��װ�������� (Install Core Requirements)
echo.
echo [��Ϣ] ���ڰ�װ�������� (NiceGUI, AI Utils)...
pip install -r requirements.txt
echo.

REM 5. GPU ѡ��˵� (GPU Selection Menu)
echo ===================================================
echo           ѡ������Ӳ����������
echo ===================================================
echo 1. NVIDIA �Կ� (CUDA 12.8) - �Ƽ� RTX ϵ��
echo 2. AMD �Կ� (ROCm) - ������ RX 6000/7000 ϵ�� (Windows)
echo 3. �� CPU (�ٶȽ�������Ϊ���û�APIģʽ)
echo.
set /p gpu_choice="���������� (1-3): "

if "%gpu_choice%"=="1" goto install_nvidia
if "%gpu_choice%"=="2" goto install_amd
if "%gpu_choice%"=="3" goto install_cpu
goto end

:install_nvidia
echo.
echo [信息] 正在为 NVIDIA RTX 50 系列安装 PyTorch (CUDA 12.8)...
echo [提示] 使用最新版本以支持 sm_120 (Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
goto end

:install_amd
echo.
echo [��Ϣ] ����Ϊ AMD ��װ PyTorch (ROCm Nightly)...
echo [��ʾ] ʹ��ͨ�õ� gfx1100 ��װ���á���������ض����Կ��ͺţ�
echo        ��鿴 README �й��� "TheRock" ��˵����
echo.
pip install --pre torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1100-all/
goto end

:install_cpu
echo.
echo [��Ϣ] ����Ϊ CPU ��װ PyTorch...
pip install torch torchvision torchaudio
goto end

:end
echo.
echo ===================================================
echo    ��װ��ɣ������� start.bat ��������
echo ===================================================
pause