import platform
import subprocess
import psutil
import os

def get_cpu_model():
    try:
        if platform.system() == "Windows":
            # Using wmic
            return subprocess.check_output("wmic cpu get name", shell=True).decode().strip().split('\n')[1]
        elif platform.system() == "Darwin":
            return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        else:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
    except:
        return platform.processor()

def get_system_stats():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
    gpu_util = 0
    vram_used = 0
    vram_total = 0
    
    try:
        # Nvidia-smi query
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        else:
            startupinfo = None
            
        res = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
            startupinfo=startupinfo
        ).decode('utf-8').strip()
        
        parts = res.split(',')
        if len(parts) >= 3:
            gpu_util = int(parts[0])
            vram_used = int(parts[1])
            vram_total = int(parts[2])
    except:
        pass
        
    return {
        "cpu": cpu,
        "ram_percent": ram.percent,
        "ram_used": ram.used / (1024**3),
        "ram_total": ram.total / (1024**3),
        "gpu": gpu_util,
        "vram_used": vram_used, 
        "vram_total": vram_total 
    }
