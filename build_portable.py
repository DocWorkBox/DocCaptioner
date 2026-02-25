import os
import sys
import shutil
import urllib.request
import zipfile
import subprocess

# 配置
PYTHON_VER = "3.10.11"
EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_VER}/python-{PYTHON_VER}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
BASE_DIR = os.getcwd()
BUILD_DIR = os.path.join(BASE_DIR, "DocCaptioner_Portable")
PYTHON_DIR = os.path.join(BUILD_DIR, "python")
PROJECT_FILES = ["web_app_ng.py", "requirements.txt", "README.md", ".gitignore", "models", "app", "ui"]

def log(msg):
    print(f"[构建] {msg}")

def download_file(url, dest):
    if os.path.exists(dest):
        log(f"文件已存在，跳过下载: {dest}")
        return
    log(f"正在下载: {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        log("下载完成")
    except Exception as e:
        log(f"下载失败: {e}")
        sys.exit(1)

def main():
    if os.path.exists(BUILD_DIR):
        log(f"清理旧构建目录: {BUILD_DIR}")
        try:
            shutil.rmtree(BUILD_DIR)
        except Exception as e:
            log(f"清理失败: {e}")
            sys.exit(1)
    
    os.makedirs(PYTHON_DIR)

    # 1. 准备 Python 环境
    zip_path = os.path.join(BASE_DIR, "python_embed.zip")
    download_file(EMBED_URL, zip_path)
    
    log("正在解压 Python Embed 环境...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(PYTHON_DIR)

    # 2. 配置 ._pth 文件
    pth_ver = PYTHON_VER.replace('.', '')[:2]
    if PYTHON_VER.startswith("3.10"): pth_ver = "310"
    pth_file = os.path.join(PYTHON_DIR, f"python{pth_ver}._pth")
    
    if os.path.exists(pth_file):
        log("配置 ._pth 文件...")
        with open(pth_file, 'r') as f:
            lines = f.readlines()
        with open(pth_file, 'w') as f:
            for line in lines:
                if line.strip() == "#import site":
                    f.write("import site\n")
                else:
                    f.write(line)

    # 3. 安装 pip
    get_pip_path = os.path.join(BASE_DIR, "get-pip.py")
    download_file(GET_PIP_URL, get_pip_path)
    
    py_exe = os.path.join(PYTHON_DIR, "python.exe")
    log("正在安装 pip...")
    subprocess.check_call([py_exe, get_pip_path], cwd=BUILD_DIR)

    # 4. 安装 PyTorch (适配 RTX 50 系列 / CUDA 12.8)
    log("正在下载并安装 PyTorch (CUDA 12.8 / Nightly)...")
    try:
        subprocess.check_call([
            py_exe, "-m", "pip", "install", 
            "torch", 
            "torchvision", 
            "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu128",
            "--no-cache-dir"
        ], cwd=BUILD_DIR)
    except Exception as e:
        log(f"PyTorch 安装失败: {e}")
        sys.exit(1)

    # 4.1 安装 bitsandbytes (Windows版)
    log("正在安装 bitsandbytes (Windows版)...")
    try:
        subprocess.check_call([
            py_exe, "-m", "pip", "install", 
            "https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py310-none-win_amd64.whl"
        ], cwd=BUILD_DIR)
    except Exception as e:
        log(f"bitsandbytes 安装失败: {e}")

    # 5. 安装其他依赖
    log("正在安装其他依赖...")
    req_path = os.path.join(BASE_DIR, "requirements.txt")
    temp_req = os.path.join(BUILD_DIR, "portable_req.txt")
    
    with open(req_path, 'r') as f:
        lines = f.readlines()
    
    with open(temp_req, 'w') as f:
        for line in lines:
            # 剔除 torch 和 llama-cpp
            if "torch" in line or "llama-cpp-python" in line:
                continue
            f.write(line)
            
    subprocess.check_call([
        py_exe, "-m", "pip", "install", 
        "-r", temp_req
    ], cwd=BUILD_DIR)

    # 6. 安装 llama-cpp-python
    log("尝试安装 llama-cpp-python (CUDA预编译版)...")
    try:
        subprocess.check_call([
            py_exe, "-m", "pip", "install", 
            "llama-cpp-python", 
            "--prefer-binary", 
            "--extra-index-url", "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels"
        ], cwd=BUILD_DIR)
    except:
        log("Llama-CPP 安装失败，将跳过 GGUF 支持。")

    # 7. 复制项目文件
    log("复制项目文件...")
    for f in PROJECT_FILES:
        src = os.path.join(BASE_DIR, f)
        dest = os.path.join(BUILD_DIR, f)
        if os.path.exists(src):
            if os.path.isdir(src):
                # 复制 models 等文件夹，排除 git
                cmd = ["robocopy", src, dest, "/E", "/XD", ".git", "__pycache__", "/NFL", "/NDL", "/NJH", "/NJS"]
                subprocess.run(cmd)
            else:
                shutil.copy2(src, dest)

    # 8. 创建 Python 启动引导脚本 (run_app.py)
    boot_script = os.path.join(BUILD_DIR, "run_app.py")
    with open(boot_script, "w", encoding="utf-8") as f:
        f.write("import sys\n")
        f.write("import os\n")
        f.write("\n")
        f.write("BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("SITE_PACKAGES = os.path.join(BASE_DIR, 'python', 'Lib', 'site-packages')\n")
        f.write("\n")
        f.write("# 1. 添加当前目录\n")
        f.write("if BASE_DIR not in sys.path:\n")
        f.write("    sys.path.insert(0, BASE_DIR)\n")
        f.write("\n")
        f.write("# 2. 手动添加 site-packages\n")
        f.write("if SITE_PACKAGES not in sys.path:\n")
        f.write("    sys.path.append(SITE_PACKAGES)\n")
        f.write("\n")
        f.write("# 3. DLL 注册 (增强版)\n")
        f.write("if hasattr(os, 'add_dll_directory'):\n")
        f.write("    try:\n")
        f.write("        os.add_dll_directory(SITE_PACKAGES)\n")
        f.write("        # Use separate components for path joining to ensure OS compatibility\n")
        f.write("        dll_libs = ['numpy.libs', 'pillow.libs', 'cv2', os.path.join('torch', 'lib')]\n")
        f.write("        for lib in dll_libs:\n")
        f.write("            lib_path = os.path.join(SITE_PACKAGES, lib)\n")
        f.write("            if os.path.exists(lib_path):\n")
        f.write("                os.add_dll_directory(lib_path)\n")
        f.write("                os.environ['PATH'] = lib_path + os.pathsep + os.environ['PATH']\n")
        f.write("    except Exception as e:\n")
        f.write("        print(f'Warning: Failed to add DLL directory: {e}')\n")
        f.write("\n")
        f.write("# 强制加入 PATH\n")
        f.write("torch_lib = os.path.join(SITE_PACKAGES, 'torch', 'lib')\n")
        f.write("if os.path.exists(torch_lib):\n")
        f.write("     os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']\n")
        f.write("\n")
        f.write("print('Starting DocCaptioner...')\n")
        f.write("try:\n")
        f.write("    import torch\n")
        f.write("    import torchvision\n")
        f.write("    print(f'Torch Version: {torch.__version__}')\n")
        f.write("    print(f'TorchVision Version: {torchvision.__version__}')\n")
        f.write("    print(f'CUDA Available: {torch.cuda.is_available()}')\n")
        f.write("    try:\n")
        f.write("        from torchvision import ops\n")
        f.write("        print('Testing NMS operator...')\n")
        f.write("        # Simple NMS test\n")
        f.write("        if torch.cuda.is_available():\n")
        f.write("            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).cuda()\n")
        f.write("            scores = torch.tensor([1.0]).cuda()\n")
        f.write("        else:\n")
        f.write("            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])\n")
        f.write("            scores = torch.tensor([1.0])\n")
        f.write("        ops.nms(boxes, scores, 0.5)\n")
        f.write("        print('NMS Operator: OK')\n")
        f.write("    except Exception as e:\n")
        f.write("        print(f'NMS Operator Failed: {e}')\n")
        f.write("except Exception as e:\n")
        f.write("    print(f'Pre-import warning: {e}')\n")
        f.write("\n")
        f.write("try:\n")
        f.write("    import web_app_ng\n")
        f.write("except ImportError as e:\n")
        f.write("    print(f'Startup Error: {e}')\n")
        f.write("    import traceback\n")
        f.write("    traceback.print_exc()\n")
        f.write("    input('Press Enter to exit...')\n")
        
    # 9. 创建 BAT 启动脚本
    start_bat = os.path.join(BUILD_DIR, "启动程序.bat")
    with open(start_bat, "w", encoding="gbk") as f:
        f.write("@echo off\n")
        f.write("title DocCaptioner Portable\n")
        f.write("cd /d %~dp0\n")
        f.write("echo Starting...\n")
        f.write("set PYTHONUNBUFFERED=1\n")
        f.write("python\\python.exe run_app.py\n")
        f.write("if %errorlevel% neq 0 (\n")
        f.write("    echo.\n")
        f.write("    echo [Error] Program crashed with exit code %errorlevel%\n")
        f.write("    pause\n")
        f.write(")\n")
        f.write("pause\n")
    
    log("="*50)
    log(f"构建完成！便携包位置: {BUILD_DIR}")
    log("="*50)

if __name__ == "__main__":
    main()
