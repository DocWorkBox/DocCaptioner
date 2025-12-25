import psutil
import subprocess
import os
import platform
import json
import asyncio
import threading
import base64
import time
import requests
import shutil
import zipfile
import hashlib
import gc
from datetime import datetime
import cv2
import numpy as np
import piexif
from PIL import Image, PngImagePlugin, ImageOps
from io import BytesIO
from nicegui import ui, app, run
from fastapi.responses import FileResponse

# 尝试导入 PyTorch 和 HuggingFace
try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from huggingface_hub import snapshot_download
except ImportError:
    torch = None

# 尝试导入 Llama-CPP (GGUF)
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

# --- 常量定义 ---
KNOWN_MODELS = {
    "Qwen3-VL-8B-Instruct (HF)": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct"},
    "Qwen3-VL-4B-Instruct (HF)": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
    "Qwen3-VL-30B-A3B-Instruct (HF)": {"repo_id": "Qwen/Qwen3-VL-30B-A3B-Instruct"},
    "Qwen3-VL-32B-Instruct-1M-Q8_0-GGUF (Unsloth)": {
        "repo_id": "unsloth/Qwen3-VL-32B-Instruct-1M-GGUF", 
        "is_gguf": True,
        "filename": "Qwen3-VL-32B-Instruct-1M-Q8_0.gguf"
    },
    "Qwen3-VL-8B-Instruct-1M-Q8_0-GGUF (Unsloth)": {
        "repo_id": "unsloth/Qwen3-VL-8B-Instruct-1M-GGUF",
        "is_gguf": True,
        "filename": "Qwen3-VL-8B-Instruct-1M-Q8_0.gguf"
    }
}

BASE_QUICK_TAGS = [
    "1girl", "1boy", "solo", "simple background", "white background",
    "masterpiece", "best quality", "highres", "absurdres",
    "monochrome", "greyscale", "sketch", "lineart",
    "outdoors", "indoors", "day", "night", "sunset",
    "upper body", "full body", "cowboy shot", "portrait",
    "looking at viewer", "smile", "blush", "blue eyes",
    "long hair", "short hair", "blonde hair", "black hair"
]

PROMPTS = {
    "🖼️ 标签生成 (Tag Generation)": "Your task is to generate a clean list of comma-separated tags for a text-to-image AI, based *only* on the visual information in the image. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "🖼️ 简单描述 (Short Description)": "Analyze the image and write a single concise sentence that describes the main subject and setting. Keep it grounded in visible details only.",
    "🖼️ 详细描述 (Detailed Description)": "Generate a detailed paragraph that combines the subject, actions, environment, lighting, and mood into 2-3 cohesive sentences. Focus on accurate visual details rather than speculation.",
    "🖼️ 超详尽描述 (Extremely Detailed)": "Produce an extremely rich description touching on appearance, clothing textures, background elements, light quality, shadows, and atmosphere. Aim for an immersive depiction rooted in what the image shows.",
    "🎬 电影感描述 (Cinematic)": "Describe the scene as if capturing a cinematic shot. Cover subject, pose, environment, lighting, mood, and artistic style (photorealistic, painterly, etc.) in one vivid paragraph emphasizing visual impact.",
    "🖼️ 详细分析 (Analysis)": "Describe this image in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "📹 视频摘要 (Video Summary)": "Summarize the key events and narrative points in this video.",
    "📖 短篇故事 (Story)": "Write a short, imaginative story inspired by this image or video.",
    "Danbooru Tags (Anime)": "Describe this image using Danbooru tags, separated by commas."
}

CONFIG_FILE = "config.json"
DATASET_ROOT = os.path.join(os.getcwd(), "Dataset Collections")
THUMB_DIR = os.path.join(os.getcwd(), "thumbnails")

if not os.path.exists(DATASET_ROOT):
    os.makedirs(DATASET_ROOT)

if not os.path.exists(THUMB_DIR):
    os.makedirs(THUMB_DIR)

# --- Thumbnail API ---
@app.get('/api/thumbnail')
def get_thumbnail(path: str):
    if not os.path.exists(path):
        return 404
    
    try:
        # Generate unique thumb name based on path + mtime (to handle updates)
        mtime = os.path.getmtime(path)
        hash_str = f"{path}_{mtime}"
        thumb_name = hashlib.md5(hash_str.encode('utf-8')).hexdigest() + ".jpg"
        thumb_path = os.path.join(THUMB_DIR, thumb_name)
        
        if not os.path.exists(thumb_path):
            try:
                # Use CV2 for Video frame extraction if it is a video
                if is_video(path):
                    cap = cv2.VideoCapture(path)
                    # Try to grab a frame from the middle or at least after 1 sec
                    # But for speed, just grab first valid frame
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame)
                    else:
                         # Failed to read video
                         raise Exception("Could not read video frame")
                else:
                    # Use PIL for images
                    img = Image.open(path)
                    # Handle rotation from EXIF if needed
                    img = ImageOps.exif_transpose(img)
                
                # Resize to max 300x300 for gallery cards
                img.thumbnail((300, 300))
                
                # Convert to RGB if needed (e.g. RGBA png to jpg)
                if img.mode in ('RGBA', 'P'): 
                    img = img.convert('RGB')
                
                img.save(thumb_path, "JPEG", quality=80)
            except Exception as e:
                print(f"Thumbnail generation failed for {path}: {e}")
                # Fallback to original if generation fails (e.g. broken image)
                # But to avoid crash, maybe return a placeholder?
                # For now, let's just return the original if thumb fails, 
                # but this might still crash if it's the original that is problematic.
                # Safer: return a 1x1 blank image or error.
                return 500

        return FileResponse(thumb_path)
    except Exception as e:
        print(f"API Error: {e}")
        return 500

# --- 样式定义 ---
CARD_STYLE = "w-full p-0 gap-0 border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow bg-white"
BTN_PRIMARY = "bg-blue-600 text-white hover:bg-blue-700 rounded-full px-4 py-2 text-sm font-medium shadow-sm transition-transform hover:-translate-y-0.5"
BTN_SECONDARY = "bg-white !text-gray-700 border border-gray-300 hover:bg-gray-50 rounded-full px-4 py-2 text-sm font-medium shadow-sm"
BTN_DANGER = "bg-red-500 text-white hover:bg-red-600 rounded-full px-4 py-2 text-sm font-medium shadow-sm"
INPUT_STYLE = "w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"

class AppState:
    def __init__(self):
        self.config = self.load_config()
        self.config.setdefault("current_folder", os.path.join(DATASET_ROOT, "default_dataset") if os.path.exists(os.path.join(DATASET_ROOT, "default_dataset")) else DATASET_ROOT)
        self.config.setdefault("source_type", "预设模型 (Preset)")
        self.config.setdefault("selected_model_key", list(KNOWN_MODELS.keys())[0])
        self.config.setdefault("api_model_name", "Qwen/Qwen3-vl-Plus")
        self.config.setdefault("api_base_url", "https://api.openai.com/v1")
        self.config.setdefault("api_key", "")
        self.config.setdefault("prompt_template", "详细描述 (Detailed)")
        self.config.setdefault("custom_prompt", "")
        self.config.setdefault("max_tokens", 512)
        self.config.setdefault("temperature", 0.7)
        self.config.setdefault("top_p", 0.9)
        self.config.setdefault("unload_model", False)
        self.config.setdefault("custom_quick_tags", [])
        self.config.setdefault("splitter_value", 40)
        self.config.setdefault("show_perf_monitor", False)
        
        self.current_files = []
        self.selected_files = set()
        self.card_refs = {}  # Store references to caption textareas {file_path: textarea_element}
        self.logs = []
        self.is_processing = False
        self.process_progress = 0.0
        self.process_status = "就绪"

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            ui.notify(f"保存配置失败: {e}", type="negative")

    def add_log(self, msg):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        line = f"[{timestamp}] {msg}"
        self.logs.append(line)
        if len(self.logs) > 100:
            self.logs.pop(0)
        
        # Real-time UI update
        if hasattr(self, 'log_ui') and self.log_ui:
            self.log_ui.push(line)
        
        # Force UI update (essential for threaded callbacks)
        # ui.update() is global, specific elements update automatically via binding usually
        # but pushing to log is client side.
        # Let's try to notify UI if needed, or print to console for debugging
        print(line) # Debug to console

state = AppState()

# --- 辅助函数 ---
def get_caption_path(img_path):
    return os.path.splitext(img_path)[0] + ".txt"

def get_caption(img_path):
    txt_path = get_caption_path(img_path)
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def save_caption(img_path, content):
    txt_path = get_caption_path(img_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)

def toggle_tag(img_path, tag, mode="追加"):
    current = get_caption(img_path)
    tags = [t.strip() for t in current.split(',') if t.strip()]
    
    if tag in tags:
        tags.remove(tag)
        action = "removed"
    else:
        if mode == "前置":
            tags.insert(0, tag)
        else:
            tags.append(tag)
        action = "added"
    
    save_caption(img_path, ", ".join(tags))
    return action

def is_video(path):
    return path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

def get_cpu_model():
    try:
        if platform.system() == "Windows":
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

# --- AI 后台逻辑 (Threaded) ---
class AIWorker(threading.Thread):
    def __init__(self, target_files, config, prompt):
        super().__init__()
        self.target_files = target_files
        self.config = config
        self.prompt = prompt
        self.should_stop = False
        self.daemon = True

    def run(self):
        state.is_processing = True
        state.process_progress = 0.0
        state.process_status = "初始化..."
        state.add_log(f"开始任务: 处理 {len(self.target_files)} 个文件")
        
        try:
            source_type = self.config["source_type"]
            state.add_log(f"当前模式: {source_type}") # Debug log
            
            model, processor = None, None

            # --- 1. 加载模型 (本地) ---
            if source_type != "在线 API (OpenAI Compatible)":
                state.process_status = "正在加载模型..."
                state.add_log(f"正在加载本地模型: {source_type}")
                
                try:
                    model_path = ""
                    if source_type == "预设模型 (Preset)":
                        model_info = KNOWN_MODELS.get(self.config["selected_model_key"])
                        if model_info:
                            model_root = os.path.join(os.getcwd(), "models")
                            repo_id = model_info["repo_id"]
                            
                            # 尝试兼容两种文件夹命名格式
                            std_name = repo_id.replace("/", "_")
                            alt_name = repo_id.split("/")[-1]
                            std_dir = os.path.join(model_root, std_name)
                            alt_dir = os.path.join(model_root, alt_name)
                            
                            model_dir = std_dir # 默认下载目标
                            
                            # 优先使用已存在的文件夹 (检测 config.json 确保不是空文件夹)
                            if os.path.exists(alt_dir) and os.path.exists(os.path.join(alt_dir, "config.json")):
                                model_path = alt_dir
                                state.add_log(f"使用已有模型: {alt_name}")
                            elif os.path.exists(std_dir) and os.path.exists(os.path.join(std_dir, "config.json")):
                                model_path = std_dir
                                state.add_log(f"使用已有模型: {std_name}")
                            else:
                                # 都不存在，开始下载
                                state.add_log(f"未检测到完整模型，开始下载: {repo_id}...")
                                try:
                                    snapshot_download(repo_id=repo_id, local_dir=std_dir)
                                    model_path = std_dir
                                except Exception as e:
                                    raise RuntimeError(f"模型下载失败: {e}")
                    else:
                        model_path = self.config.get("local_model_path", "")

                    if not model_path or not os.path.exists(model_path):
                         raise RuntimeError("无效的模型路径")

                    if self.config.get("selected_model_key", "").endswith("GGUF") or model_path.endswith(".gguf"):
                        if not HAS_GGUF: raise RuntimeError("未安装 llama-cpp-python")
                        from llama_cpp import Llama
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        chat_handler = Llava15ChatHandler(clip_model_path=os.path.dirname(model_path))
                        # Offload layers to GPU as much as possible, but respect limit
                        model = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=2048, n_gpu_layers=-1)
                    else:
                        if torch is None: raise RuntimeError("未安装 PyTorch/Transformers")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        # VRAM Safety Check
                        if device == "cuda":
                            vram_total = torch.cuda.get_device_properties(0).total_memory
                            vram_gb = vram_total / (1024**3)
                            state.add_log(f"Detected VRAM: {vram_gb:.1f} GB")
                            
                            # Simple heuristic: 8B model in FP16 needs ~16GB. If < 16GB, warn or error.
                            # But we let it try, relying on swap if needed, though that causes freeze.
                            # Better: Load in 4-bit if possible? We don't have bitsandbytes confirmed.
                            # Just log warning for now.
                            if vram_gb < 12 and "8B" in model_path:
                                state.add_log("⚠️ 警告: 显存可能不足 (<12GB)，建议使用 API 模式或 GGUF 版本")

                        try:
                            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                            model = AutoModelForImageTextToText.from_pretrained(
                                model_path, device_map=device, torch_dtype=torch.float16 if device=="cuda" else torch.float32, trust_remote_code=True
                            ).eval()
                        except RuntimeError as e:
                            if "no kernel image is available" in str(e):
                                raise RuntimeError("您的显卡 (RTX 50系列/其他) 与当前 PyTorch 版本不兼容。请使用 API 模式。")
                            raise e

                    state.add_log("模型加载成功")
                except Exception as e:
                    state.add_log(f"模型加载错误: {e}")
                    state.is_processing = False
                    return

            # --- 2. 处理循环 ---
            for i, fpath in enumerate(self.target_files):
                if self.should_stop:
                    state.add_log("用户停止任务")
                    break
                
                fname = os.path.basename(fpath)
                state.process_status = f"正在处理: {fname} ({i+1}/{len(self.target_files)})"
                state.process_progress = (i + 1) / len(self.target_files)
                
                try:
                    result_text = ""
                    is_vid = is_video(fpath)
                    
                    # --- API 模式 ---
                    if source_type == "在线 API (OpenAI Compatible)":
                        # URL 处理: 防止重复拼接 /chat/completions
                        api_base = self.config["api_base_url"].rstrip("/")
                        if api_base.endswith("/chat/completions"):
                            api_url = api_base
                        else:
                            api_url = f"{api_base}/chat/completions"
                        
                        headers = {"Authorization": f"Bearer {self.config['api_key']}", "Content-Type": "application/json"}
                        
                        # DashScope/Qwen Native 格式支持 (简单判断 model name 或用户配置)
                        # 这里我们根据 model name 简单判断，或者默认走 OpenAI
                        is_dashscope = "qwen" in self.config["api_model_name"].lower() and "plus" in self.config["api_model_name"].lower() and "vl" in self.config["api_model_name"].lower()
                        # 更好的方式是在 UI 增加 format 选择，目前暂且自动回落或默认 OpenAI
                        
                        # 构造内容
                        if is_dashscope and "dashscope" in api_base: # 只有当显式使用 dashscope SDK 或 URL 时才需要特定格式，这里假设 OpenAI 兼容层更通用
                             # 如果用户用的是 OpenAI 兼容接口（如 vllm/ollama/OneAPI），通常都支持标准格式
                             pass
                        
                        # 标准 OpenAI Vision 格式
                        content_list = [{"type": "text", "text": self.prompt}]
                        
                        if is_vid:
                            # 视频抽帧
                            try:
                                cap = cv2.VideoCapture(fpath)
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                if total_frames > 0:
                                    indices = range(0, total_frames, 8)
                                    if len(indices) > 64: indices = np.linspace(0, total_frames-1, 64, dtype=int)
                                    for idx in indices:
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                                        ret, frame = cap.read()
                                        if ret:
                                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            buf = BytesIO()
                                            Image.fromarray(frame_rgb).save(buf, format="JPEG")
                                            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                                cap.release()
                            except Exception as e:
                                state.add_log(f"视频抽帧失败: {e}")
                        else:
                            with open(fpath, "rb") as img_f:
                                b64 = base64.b64encode(img_f.read()).decode('utf-8')
                            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        
                        payload = {
                            "model": self.config["api_model_name"],
                            "messages": [{"role": "user", "content": content_list}],
                            "max_tokens": self.config["max_tokens"]
                        }
                        
                        state.add_log(f"正在请求 API: {api_url} ...")
                        try:
                            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
                            if resp.status_code == 200:
                                result_text = resp.json()['choices'][0]['message']['content']
                                state.add_log(f"API 请求成功")
                            else:
                                state.add_log(f"API Error [{resp.status_code}]: {resp.text[:200]}")
                        except Exception as req_err:
                            state.add_log(f"API 请求异常: {req_err}")
                    
                    # --- 本地模式 ---
                    elif model:
                        if processor: # Transformers
                            if is_vid:
                                messages = [{"role": "user", "content": [{"type": "video", "video": fpath}, {"type": "text", "text": self.prompt}]}]
                            else:
                                messages = [{"role": "user", "content": [{"type": "image", "image": fpath}, {"type": "text", "text": self.prompt}]}]
                                
                            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            image_inputs, video_inputs = process_vision_info(messages)
                            
                            # 确保 inputs 也在正确的 device 上
                            inputs = processor(
                                text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt"
                            ).to(model.device)

                            # 增加 streamer 以便实时输出（可选），或至少在生成前 log 一下
                            state.add_log(f"开始推理: {fname}...")
                            
                            with torch.no_grad():
                                # Use autocast for mixed precision to save memory/speed up
                                with torch.autocast("cuda"):
                                    generated_ids = model.generate(**inputs, max_new_tokens=self.config["max_tokens"])
                                
                            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                            result_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                            state.add_log(f"推理完成: {fname}")
                        else: # GGUF
                             content_list = [{"type": "text", "text": self.prompt}]
                             
                             if is_vid:
                                 # GGUF 视频抽帧逻辑
                                 try:
                                     cap = cv2.VideoCapture(fpath)
                                     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                     if total_frames > 0:
                                         indices = range(0, total_frames, 8)
                                         # GGUF 限制 16 帧以防 OOM
                                         if len(indices) > 16: indices = np.linspace(0, total_frames-1, 16, dtype=int)
                                         
                                         for idx in indices:
                                             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                                             ret, frame = cap.read()
                                             if ret:
                                                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                                 buf = BytesIO()
                                                 Image.fromarray(frame_rgb).save(buf, format="JPEG")
                                                 b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                                 content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                                     cap.release()
                                 except Exception as e:
                                     state.add_log(f"GGUF视频抽帧失败: {e}")
                             else:
                                 with open(fpath, "rb") as img_f:
                                     b64 = base64.b64encode(img_f.read()).decode('utf-8')
                                 content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                             
                             messages = [{"role": "user", "content": content_list}]
                             response = model.create_chat_completion(messages=messages, max_tokens=self.config["max_tokens"])
                             result_text = response['choices'][0]['message']['content']

                    if result_text:
                        save_caption(fpath, result_text)
                        state.add_log(f"完成: {fname}")
                    
                except Exception as e:
                    state.add_log(f"处理失败 {fname}: {e}")
                
                # --- Critical: Resource Cleanup & Rate Limiting ---
                # Force cleanup to prevent VRAM fragmentation/OOM leading to system freeze
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                
                # Sleep briefly to allow system UI/rendering to catch up and GPU to cool down
                # This prevents "deadlock" feeling on the desktop
                time.sleep(0.5) 

        except Exception as e:
            state.add_log(f"任务异常: {e}")
            print(f"Worker Exception: {e}")
        finally:
            state.is_processing = False
            state.process_status = "任务完成" if not self.should_stop else "任务已停止"
            if not self.should_stop:
                state.process_progress = 1.0
                
            # 刷新画廊以显示新的 caption
            # 注意：由于这是后台线程，我们不能直接调用 refresh_gallery() (包含 UI 操作)
            # 我们只需要设置一个标志位或通知，但最简单的是利用 timer 轮询来触发一次刷新
            # 或者我们在这里更新文件列表的内存状态，然后 UI 自动更新
            # 但 refresh_gallery() 是重建 UI，必须在主线程。
            # 解决方案：使用 ui.timer 在主线程回调中执行刷新
            
            # 尝试卸载模型
            if self.config.get("unload_model", False) and source_type != "在线 API (OpenAI Compatible)":
                del model
                del processor
                if torch: torch.cuda.empty_cache()
                state.add_log("模型已卸载")

worker = None

# --- UI 界面构建 ---
# --- NiceGUI App Setup ---
app.add_static_files('/datasets', DATASET_ROOT)

@ui.page('/')
def main_page():
    ui.page_title('DocCaptioner v1.0')
    
    # 状态：当前选中的图片路径 (用于高亮标签)
    current_active_file = None 
    
    # 清理旧的 UI 引用 (页面刷新后这些引用已失效)
    state.card_refs.clear()
    
    # 确保 file_timestamps 存在 (用于图片缓存控制)
    if not hasattr(state, 'file_timestamps'):
        state.file_timestamps = {}
    
    # 数据集管理页面的选中状态
    state.selected_ds_manage = None
    
    # --- 辅助函数定义 (提前定义以供 UI 调用) ---

    # Define holder for late-binding UI refresh functions
    ui_callbacks = {
        "refresh_details": lambda: None,
        "refresh_tags_ui": lambda: None
    }

    def show_full_image(f_path):
        """显示原图或视频预览"""
        with ui.dialog() as d, ui.card().classes('w-full h-full max-w-none p-0 bg-black'):
            # Close button
            ui.button(icon='close', on_click=d.close).classes('absolute top-4 right-4 z-50 bg-black/50 text-white rounded-full hover:bg-black/80')
            
            # Use raw path with timestamp to avoid cache
            import time
            ts = int(time.time() * 1000)
            rel_path = os.path.relpath(f_path, DATASET_ROOT)
            rel_path = rel_path.replace('\\', '/')
            from urllib.parse import quote
            url_path = f'/datasets/{quote(rel_path)}?t={ts}'
            
            if is_video(f_path):
                ui.video(url_path).classes('w-full h-full').props('controls autoplay').style('object-fit: contain;')
            else:
                ui.image(url_path).classes('w-full h-full object-contain')
        d.open()

    def create_image_card(f_path):
        is_sel = f_path in state.selected_files
        border_class = "border-2 border-blue-500" if is_sel else "border border-gray-200"
        
        # 优化：复用 timestamp
        import time
        ts = int(time.time() * 1000)
        
        # 如果文件不在缓存中，更新 ts
        # 注意：这里我们只在初次加载或文件真正改变时才更新 ts
        # refresh_gallery 会重建 UI，但我们希望保持 ts 不变，除非是 force 刷新
        if f_path not in state.file_timestamps:
            state.file_timestamps[f_path] = ts
        
        current_ts = state.file_timestamps[f_path]
        
        # 构建相对 URL (用于视频等，或者点击大图)
        rel_path = os.path.relpath(f_path, DATASET_ROOT)
        rel_path = rel_path.replace('\\', '/')
        from urllib.parse import quote
        url_path = f'/datasets/{quote(rel_path)}?t={current_ts}'
        
        # 构建缩略图 URL (用于画廊显示)
        # 注意：path 参数需要传递绝对路径
        thumb_url = f'/api/thumbnail?path={quote(f_path)}'
        
        with ui.card().classes(f'w-full sm:w-48 md:w-56 lg:w-64 shrink-0 p-0 gap-0 rounded-lg shadow-sm hover:shadow-md transition-all bg-white {border_class} group') as card:
            # 媒体预览
            # 统一使用缩略图显示，避免大量视频标签导致 GPU OOM
            # 视频支持原地播放 (Click-to-Play)，图片点击放大
            with ui.element('div').classes('relative w-full aspect-square bg-gray-100') as media_wrapper:
                def play_video_inplace():
                    media_wrapper.clear()
                    with media_wrapper:
                        ui.video(url_path).classes('w-full h-full bg-black').props('controls autoplay').style('object-fit: contain;')

                if is_video(f_path):
                    ui.image(thumb_url).classes('w-full h-full object-cover').props('loading="lazy"')
                    # Video Overlay: Play Icon
                    ui.icon('play_circle_outline').classes('absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white text-5xl opacity-80 group-hover:scale-110 transition-transform pointer-events-none drop-shadow-md')
                    # Click Overlay
                    ui.element('div').classes('absolute inset-0 cursor-pointer').on('click', play_video_inplace)
                else:
                    # Image Overlay: Zoom Icon
                    ui.image(thumb_url).classes('w-full h-full object-cover').props('loading="lazy"')
                    ui.icon('zoom_in').classes('absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white text-4xl opacity-0 group-hover:opacity-80 transition-opacity pointer-events-none drop-shadow-md')
                    # Click Overlay
                    ui.element('div').classes('absolute inset-0 cursor-pointer').on('click', lambda: show_full_image(f_path))
            
            with ui.column().classes('p-2 w-full gap-2'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(os.path.basename(f_path)[:15]+'...').classes('text-xs text-gray-500 truncate').props(f'title="{os.path.basename(f_path)}"')
                    
                    def on_chk(e, p=f_path, c=card):
                        toggle_select(e.value, p)
                        # 直接更新样式，不需要重建
                        if e.value: c.classes(remove="border border-gray-200", add="border-2 border-blue-500")
                        else: c.classes(remove="border-2 border-blue-500", add="border border-gray-200")
                        # refresh_quick_tags() # 可选，实时更新标签高亮
                    
                    chk = ui.checkbox(on_change=lambda e: on_chk(e))
                    chk.value = is_sel
                
                cap_val = get_caption(f_path)
                txt = ui.textarea(value=cap_val, on_change=lambda e: save_caption(f_path, e.value)).props('rows=3 borderless dense autogrow').classes('w-full text-xs bg-gray-50 rounded p-1')
                
                # 存储完整引用
                state.card_refs[f_path] = {"txt": txt, "chk": chk, "card": card} 
 

    def toggle_select(val, path):
        if val: state.selected_files.add(path)
        else: state.selected_files.discard(path)
        count = len(state.selected_files)
        selected_count_label.set_text(f"已选 {count}")
        refresh_quick_tags()
        # Call safe wrapper
        ui_callbacks["refresh_details"]()

    def toggle_all(val):
        folder = state.config["current_folder"]
        if val:
            for f in state.current_files:
                state.selected_files.add(os.path.join(folder, f))
        else:
            state.selected_files.clear()
            
        # 优化：不重建画廊，只更新 Checkbox 和边框样式
        for f_path, refs in state.card_refs.items():
            if isinstance(refs, dict):
                chk = refs.get("chk")
                card = refs.get("card")
                if chk and card:
                    is_sel = f_path in state.selected_files
                    # 更新 checkbox 值 (不需要触发 on_change 事件，因为我们已经更新了 selected_files)
                    # 注意：chk.value = is_sel 会触发 on_change，导致重复调用 toggle_select
                    # 我们可以临时禁用 on_change，或者在 on_chk 中判断
                    # 这里直接设值，副作用是可以接受的（add/discard 是幂等的）
                    chk.value = is_sel
                    
                    # 更新边框
                    if is_sel: 
                        card.classes(remove="border border-gray-200", add="border-2 border-blue-500")
                    else: 
                        card.classes(remove="border-2 border-blue-500", add="border border-gray-200")
        
        count = len(state.selected_files)
        selected_count_label.set_text(f"已选 {count}")
        
        refresh_quick_tags()
        # Call safe wrapper
        ui_callbacks["refresh_details"]()


    def refresh_captions_inplace():
        for fpath, refs in state.card_refs.items():
            if isinstance(refs, dict):
                txt_elem = refs.get("txt")
            else:
                txt_elem = refs # Fallback for old structure
                
            if txt_elem:
                new_val = get_caption(fpath)
                if txt_elem.value != new_val:
                    txt_elem.value = new_val

    def refresh_gallery(keep_selection=False, force=False):
        """刷新当前画廊和文件列表"""
        
        # 刷新数据集下拉列表 (如果 ds_select 存在)
        if hasattr(state, 'ds_select') and state.ds_select:
            try:
                new_ds_list = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
                # 保持当前值（如果仍然有效），否则重置
                current_val = state.ds_select.value
                state.ds_select.options = new_ds_list
                state.ds_select.update()
                # 如果当前选择的文件夹被删除了，可能需要处理，但一般 refresh 是为了看到新增
            except Exception as ex:
                print(f"Dataset list refresh failed: {ex}")

        folder = state.config["current_folder"]
        if not os.path.exists(folder):
            ui.notify(f"文件夹不存在: {folder}")
            return

        try:
            exts = ('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov', '.mkv')
            current_files_stat = {}
            for f in os.listdir(folder):
                if f.lower().endswith(exts):
                    full_path = os.path.join(folder, f)
                    txt_path = get_caption_path(full_path)
                    mtime = os.path.getmtime(full_path)
                    txt_mtime = os.path.getmtime(txt_path) if os.path.exists(txt_path) else 0
                    current_files_stat[f] = (mtime, txt_mtime)
            
            new_files = sorted(list(current_files_stat.keys()))
            files_changed = new_files != state.current_files
            
            # 检测文件是否被修改 (不仅仅是列表变化，还包括 mtime 变化)
            # 如果 mtime 变了，我们需要更新 file_timestamps 以刷新图片
            if hasattr(state, 'last_files_stat') and state.last_files_stat:
                for f, stats in current_files_stat.items():
                    full_path = os.path.join(folder, f)
                    if f in state.last_files_stat:
                        old_stats = state.last_files_stat[f]
                        # 比较图片 mtime (stats[0])
                        if stats[0] != old_stats[0]:
                            # 图片被修改，更新 timestamp
                            import time
                            if hasattr(state, 'file_timestamps'):
                                state.file_timestamps[full_path] = int(time.time() * 1000)
            
            captions_changed = False
            if not files_changed:
                last_stats = getattr(state, 'last_files_stat', {})
                for f, stats in current_files_stat.items():
                    if f in last_stats and stats != last_stats[f]:
                        captions_changed = True
                        break
                if not last_stats: captions_changed = True

            state.last_files_stat = current_files_stat

            if force or files_changed:
                state.current_files = new_files
                
                if gallery_container:
                    gallery_container.clear()
                    state.card_refs.clear()
                    
                    if not keep_selection:
                        state.selected_files.clear()
                    
                    with gallery_container:
                        for f in state.current_files[:100]:
                            f_path = os.path.join(folder, f)
                            create_image_card(f_path)
            
            elif captions_changed:
                refresh_captions_inplace()

            count = len(state.selected_files)
            selected_count_label.set_text(f"已选 {count}")

        except Exception as e:
            print(f"Refresh Error: {e}")

    def delete_selected():
        if not state.selected_files: return
        for f in list(state.selected_files):
            try:
                os.remove(f)
                txt = get_caption_path(f)
                if os.path.exists(txt): os.remove(txt)
            except Exception as e:
                ui.notify(f"删除失败 {os.path.basename(f)}: {e}", type="negative")
        refresh_gallery()
        ui.notify("已删除选中文件")

    def create_dataset():
        name = new_col_name.value
        if not name: return
        path = os.path.join(DATASET_ROOT, name)
        if not os.path.exists(path):
            os.makedirs(path)
            refresh_ds_list()
            ui.notify(f"已创建集合: {name}")
        else:
            ui.notify("集合已存在", type="warning")

    # Metadata State
    current_meta_path = {"path": None, "is_png": False, "info": {}}

    def load_metadata_ui(f_path):
        current_meta_path["path"] = f_path
        meta_preview.set_source(f_path)
        meta_input.value = ""
        rows = []
        
        try:
            if f_path.lower().endswith('.png'):
                current_meta_path["is_png"] = True
                img = Image.open(f_path)
                img.load()
                current_meta_path["info"] = img.info
                for k, v in img.info.items():
                    rows.append({'key': k, 'value': str(v)})
                    if k in ["parameters", "UserComment", "Description"]:
                        meta_input.value = str(v)
            else: # JPG/Other
                current_meta_path["is_png"] = False
                try:
                    exif_dict = piexif.load(f_path)
                    current_meta_path["info"] = exif_dict
                    # Parse EXIF (Simplified)
                    for ifd in ("0th", "Exif", "GPS", "1st"):
                        if ifd in exif_dict:
                            for tag, val in exif_dict[ifd].items():
                                name = piexif.TAGS[ifd].get(tag, {"name": str(tag)})["name"]
                                rows.append({'key': f"{ifd}-{name}", 'value': str(val)[:100]})
                                if name == "UserComment":
                                    try:
                                        # Try decode
                                        if isinstance(val, bytes):
                                            if val.startswith(b'UNICODE'): meta_input.value = val[8:].decode('utf-16').strip('\x00')
                                            elif val.startswith(b'ASCII'): meta_input.value = val[5:].decode('ascii').strip('\x00')
                                    except: pass
                except: pass
        except Exception as e:
            ui.notify(f"读取元数据失败: {e}", type="negative")
        
        meta_table.rows = rows
        
    def save_metadata():
        f_path = current_meta_path["path"]
        if not f_path or not os.path.exists(f_path): return
        
        try:
            if current_meta_path["is_png"]:
                target_info = PngImagePlugin.PngInfo()
                for r in meta_table.rows:
                    target_info.add_text(r['key'], r['value'])
                # Override parameters
                if meta_input.value:
                    target_info.add_text("parameters", meta_input.value)
                
                img = Image.open(f_path)
                img.save(f_path, pnginfo=target_info)
            else:
                # Simple UserComment update for JPG
                exif_dict = piexif.load(f_path)
                if meta_input.value:
                    user_comment = b'UNICODE\x00' + meta_input.value.encode('utf-16le')
                    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, f_path)
            
            ui.notify("元数据已保存")
        except Exception as e:
            ui.notify(f"保存失败: {e}", type="negative")

    def strip_metadata():
        f_path = current_meta_path["path"]
        if not f_path: return
        try:
            img = Image.open(f_path)
            data = list(img.getdata())
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(data)
            clean_img.save(f_path)
            ui.notify("元数据已清除")
            load_metadata_ui(f_path)
        except Exception as e:
            ui.notify(f"清除失败: {e}", type="negative")

    def start_batch_process():
        if not state.selected_files:
            ui.notify("请先选择图片", type="warning")
            return
            
        op = operation.value
        ui.notify(f"开始批量处理: {op}")
        
        # 确定输出目录
        target_dir = os.path.dirname(list(state.selected_files)[0]) # 默认当前目录
        is_new_folder = output_new_folder_chk.value
        if is_new_folder:
            folder_name = output_folder_name.value.strip()
            if not folder_name:
                ui.notify("请输入新文件夹名称", type="warning")
                return
            target_dir = os.path.join(DATASET_ROOT, folder_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
        elif not is_new_folder and "Image Edits" in target_dir:
            # 防止默认行为导致的误解，如果未勾选新文件夹，则不应该有 Image Edits 文件夹
            # 除非当前目录就是 Image Edits
            pass

        
        # 准备数据
        files_to_process = sorted(list(state.selected_files))
        total_files = len(files_to_process)
        count = 0
        
        # 智能位数计算 (用于重命名)
        if total_files < 100: z_fill = 2
        elif total_files < 1000: z_fill = 3
        else: z_fill = 4
        
        rename_prefix = rename_prefix_input.value.strip()
        
        try:
            for idx, f in enumerate(files_to_process):
                fname = os.path.basename(f)
                base_name, ext = os.path.splitext(fname)
                
                # --- 操作逻辑 ---
                if op == "顺序重命名 (Rename)":
                    new_name = f"{rename_prefix}_{str(idx+1).zfill(z_fill)}{ext}"
                    dest_path = os.path.join(target_dir, new_name)
                    
                    # 复制或移动
                    if is_new_folder:
                        shutil.copy2(f, dest_path)
                        # 同时复制 txt (如果存在)
                        txt_src = get_caption_path(f)
                        if os.path.exists(txt_src):
                            txt_dest = os.path.splitext(dest_path)[0] + ".txt"
                            shutil.copy2(txt_src, txt_dest)
                    else:
                        # 原地重命名
                        os.rename(f, dest_path)
                        txt_src = get_caption_path(f)
                        if os.path.exists(txt_src):
                            txt_dest = os.path.splitext(dest_path)[0] + ".txt"
                            os.rename(txt_src, txt_dest)
                            
                else:
                    # 图片处理操作 (Resize/Crop/Rotate/Convert)
                    # 确保正确读取图片
                    img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None: 
                        print(f"无法读取图片: {f}")
                        continue
                    
                    res = img
                    if op == "调整大小 (Resize)":
                        h, w = img.shape[:2]
                        if resize_mode.value == "按长边缩放":
                            target = int(resize_size.value)
                            scale = target / max(h, w)
                            res = cv2.resize(img, (0,0), fx=scale, fy=scale)
                        else:
                            res = cv2.resize(img, (int(resize_w.value), int(resize_h.value)))
                    elif op == "旋转 (Rotate)":
                        # 注意：Rotate 参数必须是 int
                        code = cv2.ROTATE_90_CLOCKWISE if rotate_dir.value == "顺时针 90°" else cv2.ROTATE_90_COUNTERCLOCKWISE
                        res = cv2.rotate(img, code)
                    elif op == "裁剪 (Crop)":
                        cw, ch = int(crop_w.value), int(crop_h.value)
                        h, w = img.shape[:2]
                        # Simple Center Crop
                        x = (w - cw) // 2
                        y = (h - ch) // 2
                        if x >= 0 and y >= 0 and x+cw <= w and y+ch <= h:
                            res = img[y:y+ch, x:x+cw]
                        else:
                             print(f"裁剪尺寸超出图片范围: {f}")
                             # 如果裁剪无效，可以选择保留原图或跳过，这里保留原图
                             res = img
                    
                    # 保存结果
                    save_ext = ext
                    if op == "转换格式 (Convert)":
                        save_ext = f".{format_select.value}"
                        # 确保 save_ext 正确替换掉旧扩展名
                        # base_name 已经是无后缀的文件名
                    
                    # 命名处理
                    if op == "转换格式 (Convert)":
                        # 转换格式时，如果不是保存到新文件夹，我们应该删除原文件吗？
                        # 用户明确要求：不要创建副本，要直接替换（如果可能）
                        # 但如果是不同扩展名，严格来说是创建新文件。
                        # 为了满足用户“不创建副本”的直观感受，我们需要在转换成功后删除原文件。
                        
                        save_name = base_name + save_ext
                        save_path = os.path.join(target_dir, save_name)
                        
                        # 检查是否是原地转换（路径相同但扩展名不同）
                        is_inplace_convert = not is_new_folder and save_path != f
                        
                    else:
                        # 其他操作，保持原扩展名
                        save_name = base_name + save_ext
                        save_path = os.path.join(target_dir, save_name)
                        is_inplace_convert = False
                    
                    params = []
                    if save_ext.lower() in ['.jpg', '.jpeg']: 
                        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality_slider.value)]
                    
                    # 使用 imencode 保存以支持中文路径
                    success, buf = cv2.imencode(save_ext, res, params)
                    if success:
                        with open(save_path, 'wb') as f_out:
                            buf.tofile(f_out)
                            
                        # 如果是原地转换格式，删除原文件
                        if is_inplace_convert:
                            try:
                                os.remove(f)
                                # 还要处理 txt 吗？txt 文件名应该已经通过下面的逻辑同步了
                                # 但旧的 txt 文件名和新的一样（因为只改了图片后缀），所以不需要删除 txt
                            except Exception as ex:
                                print(f"无法删除原文件: {f}, {ex}")
                    
                    # 总是尝试复制/同步 txt
                    # ...
                    
                    # 总是尝试复制 txt (无论是否是新文件夹，或者是否改名)
                    txt_src = get_caption_path(f)
                    if os.path.exists(txt_src):
                        txt_dest = os.path.splitext(save_path)[0] + ".txt"
                        if os.path.abspath(txt_src) != os.path.abspath(txt_dest):
                            shutil.copy2(txt_src, txt_dest)

                count += 1
            
            ui.notify(f"批量处理完成: {count} 张")
            
            # --- 后处理刷新 ---
            if is_new_folder:
                # ... (原有逻辑不变)
                new_ds_list = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
                if hasattr(state, 'ds_select'):
                    state.ds_select.options = new_ds_list
                    state.ds_select.value = folder_name 
                ui.notify(f"新数据集已创建并切换: {folder_name}", type="positive")
            else:
                # 原地修改，强制刷新画廊
                refresh_gallery(force=True)
            
            # 无论如何，重置全选框状态 (因为刷新后 selection 被清空)
            if hasattr(state, 'select_all_checkbox') and state.select_all_checkbox:
                state.select_all_checkbox.value = False
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            ui.notify(f"批量处理出错: {e}", type="negative")
            print(f"Batch Error: {e}")

    # --- 顶部导航栏 ---
    with ui.header().classes('bg-white text-gray-800 border-b border-gray-200 h-20 px-4 flex items-center justify-between shadow-sm z-50'):
        # Left: Title
        with ui.row().classes('items-center gap-2'):
            ui.button(icon='menu', on_click=lambda: left_drawer.toggle()).classes('text-gray-600')
            ui.icon('local_offer', size='md', color='blue-600')
            with ui.column().classes('gap-0'):
                ui.label('DocCaptioner').classes('text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 leading-none')
                ui.label('v1.0').classes('text-[10px] text-gray-400 leading-none')

        # Right: Dataset & Actions
        with ui.row().classes('items-center gap-4'):
            # --- 性能监控 (New) ---
            with ui.row().classes('items-center gap-3 mr-2 border-r pr-4 border-gray-300').bind_visibility_from(state.config, 'show_perf_monitor'):
                 # CPU
                with ui.row().classes('items-center gap-1'):
                    ui.label('CPU').classes('text-xs font-bold text-black')
                    with ui.element('div').classes('relative w-16 h-4 bg-gray-200 rounded overflow-hidden'):
                        cpu_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full')
                        cpu_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[10px] font-bold text-black z-10')

                # RAM
                with ui.row().classes('items-center gap-1'):
                    ui.label('RAM').classes('text-xs font-bold text-black')
                    with ui.element('div').classes('relative w-16 h-4 bg-gray-200 rounded overflow-hidden'):
                        ram_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full text-purple-500').props('color=purple')
                        ram_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[10px] font-bold text-black z-10')

                # GPU
                with ui.row().classes('items-center gap-1'):
                    ui.label('GPU').classes('text-xs font-bold text-black')
                    with ui.element('div').classes('relative w-16 h-4 bg-gray-200 rounded overflow-hidden'):
                        gpu_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full text-green-500').props('color=green')
                        gpu_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[10px] font-bold text-black z-10')

                # VRAM
                with ui.row().classes('items-center gap-1'):
                    ui.label('VRAM').classes('text-xs font-bold text-black')
                    with ui.element('div').classes('relative w-16 h-4 bg-gray-200 rounded overflow-hidden'):
                        vram_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full text-orange-500').props('color=orange')
                        vram_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[10px] font-bold text-black z-10')
                
                def update_perf_header():
                    if not state.config.get("show_perf_monitor"): return
                    s = get_system_stats()
                    
                    cpu_label.text = f"{s['cpu']}%"
                    cpu_bar.value = s['cpu'] / 100.0
                    
                    ram_pct = int(s['ram_percent'])
                    ram_label.text = f"{ram_pct}%"
                    ram_bar.value = ram_pct / 100.0
                    
                    gpu_label.text = f"{s['gpu']}%"
                    gpu_bar.value = s['gpu'] / 100.0
                    
                    vram_pct = s['vram_used'] / s['vram_total'] if s['vram_total'] > 0 else 0
                    vram_label.text = f"{int(vram_pct*100)}%"
                    vram_bar.value = vram_pct

                ui.timer(2.0, update_perf_header)

            # 数据集选择器美化
            with ui.row().classes('items-center gap-1 bg-gray-100 rounded-lg px-2 py-1 border border-gray-200'):
                ui.icon('folder_open', color='gray-500').classes('text-sm')
                # 将 ds_select 存储在 state 中以便全局访问
                state.ds_select = ui.select(
                    options=[d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))],
                    value=os.path.basename(state.config["current_folder"]),
                    on_change=lambda e: change_dataset(e.value)
                ).classes('w-40 text-sm').props('dense borderless options-dense behavior="menu"') # borderless 融入背景
            
            ui.button(icon='refresh', on_click=lambda: refresh_gallery(force=True)).props('flat round dense').classes('text-gray-600 hover:bg-gray-100')

    # --- 左侧侧边栏 (标签 & 控制) ---
    with ui.left_drawer(value=True).classes('bg-gray-50 border-r border-gray-200 p-4 overflow-y-auto w-80') as left_drawer:
        ui.label('🏷️ 快速标签').classes('text-lg font-bold mb-4 text-gray-700')
        
        # 标签模式
        with ui.row().classes('w-full mb-4 bg-white p-2 rounded border border-gray-200'):
            tag_mode = ui.radio(['追加', '前置'], value='追加').props('inline dense')
        
        # 标签按钮区域
        quick_tags_container = ui.row().classes('gap-2 flex-wrap mb-6')
        tag_buttons = {} # 存储按钮引用以便更新样式

        def refresh_quick_tags():
            quick_tags_container.clear()
            tag_buttons.clear()
            
            # 合并基础标签和自定义标签
            all_tags = BASE_QUICK_TAGS + state.config["custom_quick_tags"]
            
            # 获取当前选中图片(如果有)的现有标签，用于高亮
            active_tags = []
            if state.selected_files:
                # 取最后选中的一个文件作为参考
                last_selected = list(state.selected_files)[-1]
                caption = get_caption(last_selected)
                active_tags = [t.strip() for t in caption.split(',')]

            with quick_tags_container:
                for tag in all_tags:
                    # 判断高亮状态
                    is_active = tag in active_tags
                    btn_class = "bg-blue-600 text-white shadow-md" if is_active else "bg-white !text-gray-700 border border-gray-300 hover:bg-gray-100"
                    
                    def on_tag_click(t=tag):
                        # ... (现有逻辑)
                        # 如果点击标签需要刷新选中状态，可以在这里调用
                        # 但目前的标签点击主要是修改 caption
                        
                        count = 0
                        for f in state.selected_files:
                            toggle_tag(f, t, tag_mode.value)
                            count += 1
                        if count > 0: 
                            ui.notify(f"已更新 {count} 个文件")
                            refresh_captions_inplace()
                            refresh_quick_tags()
                        else:
                            ui.notify("请先选择文件", type="warning")

                    # 这里不需要重新定义 click_tag，使用上面的 on_tag_click
                    btn = ui.button(tag, on_click=on_tag_click).classes(f'px-3 py-1 text-xs rounded-full transition-all duration-200 {btn_class}')
                    tag_buttons[tag] = btn
        
        refresh_quick_tags()
        
        ui.separator().classes('my-4')
        
        # 自定义标签输入
        ui.label('自定义标签').classes('font-bold text-gray-700 mb-2')
        custom_tag_input = ui.input('输入标签...').classes(INPUT_STYLE).props('dense outlined')
        
        with ui.row().classes('w-full gap-2 mt-2'):
            def add_custom_tag_action():
                t = custom_tag_input.value
                if t:
                    count = 0
                    for f in state.selected_files:
                        toggle_tag(f, t, tag_mode.value)
                        count += 1
                    if count > 0: 
                        ui.notify(f"已更新 {count} 个文件")
                        refresh_captions_inplace()
                        refresh_quick_tags()
                    else: ui.notify("请先选择文件", type="warning")
            ui.button('应用', on_click=add_custom_tag_action).classes(BTN_SECONDARY + ' flex-1')
            
            def save_custom_tag_setting():
                t = custom_tag_input.value
                if t and t not in state.config["custom_quick_tags"]:
                    state.config["custom_quick_tags"].append(t)
                    state.save_config()
                    refresh_quick_tags()
                    ui_callbacks["refresh_tags_ui"]()
                    ui.notify("已添加到常用列表")
            ui.button('保存预设', icon='save', on_click=save_custom_tag_setting).classes(BTN_SECONDARY)
            
        # --- 性能监视器面板 (已移动到顶部) ---


    # --- 主界面布局 (左侧画廊 + 右侧功能区) ---
    # 使用 calc(100vh - 5rem) 确保分割器高度固定为视口高度减去头部高度，避免整个页面滚动
    # 头部高度 h-20 对应 5rem
    with ui.splitter(value=state.config["splitter_value"], limits=(20, 80), 
                     on_change=lambda e: update_config("splitter_value", e.value)).classes('w-full h-[calc(100vh-5rem)]') as splitter:
        
        # --- 左侧：画廊列表 (始终显示) ---
        with splitter.before:
            with ui.column().classes('w-full h-full bg-gray-100 overflow-y-auto'):
                with ui.row().classes('w-full items-center justify-between sticky top-0 bg-gray-100 z-50 py-2 px-4 border-b border-gray-200'):
                    ui.label('📸 图片/视频列表').classes('text-lg font-bold text-gray-700')
                    with ui.row().classes('gap-2 items-center'):
                        selected_count_label = ui.label("已选 0").classes('text-xs text-gray-500')
                        # 绑定 state.select_all_checkbox 以便在需要时手动重置
                        # 修复全选框不更新的问题：
                        # 在 refresh_gallery 中，我们虽然更新了 selected_count_label，但如果是在 toggle_all 内部触发的 UI 更新
                        # 实际上不需要额外操作。
                        # 但如果是外部操作（如单个取消选择），我们需要同步全选框的状态吗？
                        # 目前逻辑是单向绑定的（全选 -> 所有），反向绑定（只要有一个没选 -> 全选取消）比较复杂且耗资源
                        # 我们保持现状，但在 refresh_gallery(force=True) 时重置全选框
                        
                        state.select_all_checkbox = ui.checkbox('全选', on_change=lambda e: toggle_all(e.value))
                
                # 画廊容器
                # 使用 Flex 布局替代 Grid，实现更好的响应式
                # gap-3: 间距
                # justify-center: 居中对齐 (可选)
                gallery_container = ui.row().classes('w-full gap-3 justify-center items-stretch p-4')
                
                # 初始化：如果已有文件缓存，尝试立即渲染 (防止刷新后白屏)
                # 注意：此时 ui.timer 尚未运行
                if state.current_files:
                     with gallery_container:
                        for f in state.current_files[:100]:
                            f_path = os.path.join(state.config["current_folder"], f)
                            create_image_card(f_path)

        # --- 右侧：功能选项卡 ---
        with splitter.after:
            with ui.column().classes('w-full h-full p-0 gap-0 overflow-hidden'):
                with ui.tabs().classes('w-full border-b border-gray-200 bg-white gap-2 justify-start pl-2 shrink-0') as tabs:
                    tab_ai = ui.tab('AI 自动标注', icon='smart_toy')
                    tab_batch = ui.tab('批量处理', icon='photo_library').classes('ml-2')
                    tab_dataset = ui.tab('数据集', icon='folder_shared').classes('ml-2')
                    tab_meta = ui.tab('详细信息', icon='info').classes('ml-2')
                    tab_settings = ui.tab('设置', icon='settings').classes('ml-2')

                with ui.tab_panels(tabs, value=tab_ai).classes('w-full flex-grow overflow-hidden p-0 bg-white'):
                    
                    # --- TAB 1: AI 标注 ---
                    with ui.tab_panel(tab_ai).classes('w-full h-full p-6 flex flex-col gap-4 overflow-y-auto'):
                        ui.label('🤖 AI 自动打标').classes('text-2xl font-bold mb-2')
                        
                        with ui.card().classes('w-full p-4 bg-gray-50 border'):
                            ui.label('模型配置').classes('font-bold text-gray-700 mb-2')
                            with ui.row().classes('w-full gap-4'):
                                ui.select(["预设模型 (Preset)", "在线 API (OpenAI Compatible)", "本地路径 (Local Path)"], 
                                        value=state.config["source_type"], label="来源",
                                        on_change=lambda e: update_config("source_type", e.value)).classes('w-1/3')
                                
                                # 动态配置区域
                                config_area = ui.row().classes('flex-1 gap-4')
                                def render_ai_config():
                                    config_area.clear()
                                    with config_area:
                                        if state.config["source_type"] == "预设模型 (Preset)":
                                            ui.select(list(KNOWN_MODELS.keys()), value=state.config["selected_model_key"],
                                                    label="选择模型", on_change=lambda e: update_config("selected_model_key", e.value)).classes('flex-1')
                                            ui.checkbox("完成后卸载", value=state.config["unload_model"],
                                                    on_change=lambda e: update_config("unload_model", e.value)).classes('mt-2')
                                        elif state.config["source_type"] == "本地路径 (Local Path)":
                                            ui.input("模型路径 (文件夹或 .gguf)", value=state.config.get("local_model_path", ""),
                                                    on_change=lambda e: update_config("local_model_path", e.value)).classes('flex-1')
                                            ui.checkbox("完成后卸载", value=state.config["unload_model"],
                                                    on_change=lambda e: update_config("unload_model", e.value)).classes('mt-2')
                                        else:
                                            ui.input("API Base URL", value=state.config["api_base_url"],
                                                    on_change=lambda e: update_config("api_base_url", e.value)).classes('flex-1')
                                            ui.input("API Key", password=True, value=state.config["api_key"],
                                                    on_change=lambda e: update_config("api_key", e.value)).classes('flex-1')
                                            ui.input("Model Name", value=state.config["api_model_name"],
                                                    on_change=lambda e: update_config("api_model_name", e.value)).classes('w-32')
                                            
                                            def test_api():
                                                api_base = state.config["api_base_url"].rstrip("/")
                                                if api_base.endswith("/chat/completions"):
                                                    api_url = api_base
                                                else:
                                                    api_url = f"{api_base}/chat/completions"
                                                
                                                ui.notify(f"正在测试: {api_url} ...", type="info")
                                                try:
                                                    # 发送一个极简的请求
                                                    headers = {"Authorization": f"Bearer {state.config['api_key']}", "Content-Type": "application/json"}
                                                    payload = {
                                                        "model": state.config["api_model_name"],
                                                        "messages": [{"role": "user", "content": "test"}],
                                                        "max_tokens": 5
                                                    }
                                                    resp = requests.post(api_url, headers=headers, json=payload, timeout=10)
                                                    if resp.status_code == 200:
                                                        ui.notify("✅ 连接成功!", type="positive")
                                                    else:
                                                        ui.notify(f"❌ 失败 [{resp.status_code}]: {resp.text[:100]}", type="negative", close_button=True, multi_line=True)
                                                except Exception as e:
                                                    ui.notify(f"❌ 异常: {e}", type="negative", close_button=True)

                                            ui.button("🔗 测试", on_click=test_api).classes('w-auto px-3 whitespace-nowrap')
                                render_ai_config()

                        with ui.card().classes('w-full p-4 bg-gray-50 border'):
                            ui.label('提示词 (Prompt)').classes('font-bold text-gray-700 mb-2')
                            
                            # Prompt State
                            # 校验 prompt_template 是否有效，如果无效（如旧配置）则重置
                            if "prompt_template" not in state.config or state.config["prompt_template"] not in PROMPTS:
                                state.config["prompt_template"] = list(PROMPTS.keys())[0]
                            if "target_lang" not in state.config: state.config["target_lang"] = "英语 (English)"
                            
                            def update_prompt_text():
                                t_key = state.config["prompt_template"]
                                t_lang = state.config["target_lang"]
                                
                                base_text = PROMPTS.get(t_key, "")
                                suffix = ""
                                if "中文" in t_lang and "双语" not in t_lang:
                                    suffix = "\n\n请直接输出中文描述，不要包含任何开场白（如“好的”、“这是一段描述”等）或结束语。"
                                elif "双语" in t_lang:
                                    suffix = (
                                        "\n\n请提供中文和英文双语描述。\n"
                                        "要求格式严格如下：\n"
                                        "## Chinese Description\n"
                                        "[中文内容]\n\n"
                                        "## English Description\n"
                                        "[English Content]\n\n"
                                        "注意：不要包含任何开场白或多余的解释性文字，直接输出内容。"
                                    )
                                else:
                                    suffix = "\n\nOutput the description directly without any conversational fillers (e.g., 'Here is a description')."
                                
                                if "自定义" not in t_key:
                                    prompt_input.value = base_text + suffix

                            with ui.row().classes('w-full gap-4'):
                                ui.select(list(PROMPTS.keys()), value=state.config["prompt_template"], label="模板",
                                          on_change=lambda e: [update_config("prompt_template", e.value), update_prompt_text()]).classes('w-1/2')
                                
                                ui.select(["英语 (English)", "中文 (Chinese)", "中英双语 (Bilingual)"], 
                                          value=state.config.get("target_lang", "英语 (English)"), label="目标语言",
                                          on_change=lambda e: [update_config("target_lang", e.value), update_prompt_text()]).classes('w-1/2')

                            prompt_input = ui.textarea(value=PROMPTS.get(state.config["prompt_template"], "")).classes('w-full h-32 bg-white')
                            
                            # User Extra Prompt
                            ui.label('用户额外提示词 (User Input / Context)').classes('text-sm text-gray-500 mt-2')
                            user_extra_prompt = ui.textarea(placeholder="输入具体要求、关键词，如：图中出现的人物以D0c来指代，不具体描述外貌细节").props('rows=2').classes('w-full bg-white')
                            
                            # Initialize
                            update_prompt_text()

                        # 进度与控制
                        ui.separator().classes('my-2')
                        with ui.row().classes('w-full items-center relative h-6'):
                            # 进度条
                            progress_bar = ui.linear_progress(value=0.0, show_value=False).classes('w-full h-6 rounded-full absolute top-0 left-0')
                            # 进度文字 (居中叠加)
                            progress_label = ui.label('0%').classes('z-10 w-full text-center text-xs font-bold text-white mix-blend-difference')

                        status_label = ui.label('就绪').classes('text-sm text-gray-500 mt-1')
                        
                        # 定时器：轮询后台状态以更新 UI (解决多线程 UI 不刷新问题)
                        def update_ui_loop():
                            progress_bar.value = state.process_progress
                            # 格式化进度显示: xx%
                            progress_label.text = f"{int(state.process_progress * 100)}%"
                            status_label.text = state.process_status
                            
                            # 检测任务刚刚完成 (从 processing=True 变为 False)
                            # 我们需要一个状态标记来避免重复刷新
                            if not state.is_processing and state.process_progress >= 1.0 and "完成" in status_label.text:
                                if not getattr(state, 'has_auto_refreshed', False):
                                    state.has_auto_refreshed = True
                                    # 强制在主线程刷新 UI，但 refresh_gallery 本身操作的是全局 UI 容器
                                    # 在 NiceGUI 中，timer 回调是在主事件循环中的，所以直接调用是安全的
                                    refresh_gallery()
                                    ui.notify("任务完成，列表已刷新", type="positive")
                                    
                                    # 强制更新容器，虽然 .clear() 应该自动处理，但加上这个更保险
                                    if gallery_container:
                                        gallery_container.update()
                            elif state.is_processing:
                                state.has_auto_refreshed = False

                        ui.timer(0.1, update_ui_loop)

                        with ui.row().classes('w-full gap-4 mt-2'):
                            def start_ai_task():
                                # 1. 优先使用用户勾选的文件
                                target_files = list(state.selected_files)
                                
                                # 2. 如果没勾选，则警告用户
                                if not target_files:
                                    ui.notify("请先在左侧选择要处理的图片/视频！", type="warning", position="center")
                                    return
                                
                                # Combine Prompt
                                final_prompt = prompt_input.value
                                if user_extra_prompt.value.strip():
                                    final_prompt += f"\n\n[User Context/Input]:\n{user_extra_prompt.value}"
                                    
                                global worker
                                worker = AIWorker(target_files, state.config, final_prompt)
                                worker.start()
                                ui.notify(f"任务已启动: 处理 {len(target_files)} 个文件")

                            def stop_worker():
                                global worker
                                if worker and worker.is_alive():
                                    worker.should_stop = True
                                    state.add_log("正在停止任务...")
                                    # 如果是 Transformers 模型生成，可能卡在 generate
                                    # 强制设置标志位，但无法强制杀线程
                                else:
                                    state.add_log("没有正在运行的任务")

                            ui.button('🚀 开始打标', on_click=start_ai_task).classes(BTN_PRIMARY + ' flex-1 text-lg h-12')
                            ui.button('⏹ 停止', on_click=stop_worker).classes(BTN_DANGER + ' w-32 h-12')

                        # 日志 (已移除)
                        # ui.label('运行日志').classes('font-bold mt-4')
                        # log_area = ui.log().classes('w-full flex-1 border rounded bg-gray-900 text-green-400 p-2 text-xs font-mono min-h-[10rem]')
                        
                        # 移除原有的日志刷新逻辑
                        state.log_ui = None 

            
                    # --- TAB 2: 批量编辑 ---
                    with ui.tab_panel(tab_batch).classes('w-full h-full p-6 overflow-y-auto'):
                        ui.label('✏️ 批量图片处理').classes('text-2xl font-bold mb-6')
                        
                        # 参数面板
                        with ui.card().classes('w-full max-w-2xl mx-auto p-6 bg-white shadow-md border'):
                            ui.label(f"已选择 {len(state.selected_files)} 张图片").classes('mb-4 text-blue-600 font-medium')
                            
                            ui.label('选择操作').classes('text-sm text-gray-500 mb-1')
                            operation = ui.select(
                                options=["转换格式 (Convert)", "调整大小 (Resize)", "旋转 (Rotate)", "裁剪 (Crop)", "顺序重命名 (Rename)"],
                                value="转换格式 (Convert)",
                                on_change=lambda e: update_batch_ui(e.value)
                            ).classes('w-full border rounded mb-4 text-lg')
                            
                            # 动态参数区域
                            param_container = ui.column().classes('w-full border p-4 rounded bg-gray-50 mb-4')
                            
                            # --- 定义参数控件 ---
                            with param_container:
                                # 1. Convert
                                convert_params = ui.column().classes('w-full')
                                with convert_params:
                                    ui.label('目标格式').classes('text-sm text-gray-500')
                                    format_select = ui.select(['jpg', 'png', 'webp'], value='jpg').classes('w-full')
                                    ui.label('质量 (Quality, 仅JPG)').classes('text-sm text-gray-500 mt-2')
                                    quality_slider = ui.slider(min=1, max=100, value=90).props('label')

                                # 2. Resize
                                resize_params = ui.column().classes('w-full hidden')
                                with resize_params:
                                    resize_mode = ui.radio(['指定尺寸', '按长边缩放'], value='按长边缩放').props('inline')
                                    with ui.row().classes('w-full gap-2 items-center'):
                                        resize_w = ui.number(label='宽', value=512, min=1).classes('w-20')
                                        resize_h = ui.number(label='高', value=512, min=1).classes('w-20')
                                        resize_size = ui.number(label='长边', value=1024, min=1).classes('w-20')
                                    
                                    def update_resize_ui(e=None):
                                        if resize_mode.value == '指定尺寸':
                                            resize_w.enable(); resize_h.enable(); resize_size.disable()
                                        else:
                                            resize_w.disable(); resize_h.disable(); resize_size.enable()
                                    resize_mode.on_value_change(update_resize_ui)
                                    update_resize_ui()

                                # 3. Rotate
                                rotate_params = ui.column().classes('w-full hidden')
                                with rotate_params:
                                    rotate_dir = ui.radio(["顺时针 90°", "逆时针 90°"], value="顺时针 90°")

                                # 4. Crop
                                crop_params = ui.column().classes('w-full hidden')
                                with crop_params:
                                    ui.label('中心裁剪尺寸').classes('text-sm text-gray-500')
                                    with ui.row():
                                        crop_w = ui.number(label='宽', value=512).classes('w-24')
                                        crop_h = ui.number(label='高', value=512).classes('w-24')

                                # 5. Rename (新增)
                                rename_params = ui.column().classes('w-full hidden')
                                with rename_params:
                                    ui.label('文件名前缀 (Prefix)').classes('text-sm text-gray-500')
                                    rename_prefix_input = ui.input(placeholder='例如: my_image').classes('w-full').props('clearable')
                                    ui.label('编号位数自动根据文件数量决定 (如: 001, 0001)').classes('text-xs text-gray-400 mt-1')

                            def update_batch_ui(op=None):
                                if op is None: op = operation.value
                                convert_params.set_visibility(op == "转换格式 (Convert)")
                                resize_params.set_visibility(op == "调整大小 (Resize)")
                                rotate_params.set_visibility(op == "旋转 (Rotate)")
                                crop_params.set_visibility(op == "裁剪 (Crop)")
                                rename_params.set_visibility(op == "顺序重命名 (Rename)")
                            
                            # 初始化一次
                            # 使用 lambda 确保 operation 存在且可访问
                            ui.timer(0.1, lambda: update_batch_ui(), once=True)

                            # 输出选项
                            output_new_folder_chk = ui.checkbox('保存到新文件夹', value=False).classes('mb-2')
                            output_folder_name = ui.input(placeholder='请输入新文件夹名').classes('w-full mb-4 hidden')
                            
                            def toggle_folder_input(e):
                                if e.value:
                                    output_folder_name.classes(remove='hidden')
                                else:
                                    output_folder_name.classes('hidden')
                                
                            output_new_folder_chk.on_value_change(lambda e: toggle_folder_input(e))
                            
                            def trigger_batch_process():
                                # 包装一下以传递正确的 UI 状态
                                start_batch_process()
                                
                            ui.button('▶ 执行批量处理', on_click=trigger_batch_process).classes(BTN_PRIMARY + ' w-full h-12 text-lg shadow-md')

                    # --- TAB 3: 数据集管理 ---
                    with ui.tab_panel(tab_dataset).classes('w-full h-full p-0'):
                        with ui.column().classes('w-full h-full p-6 items-stretch gap-6'):
                            ui.label('📂 数据集管理').classes('text-2xl font-bold shrink-0')
                        
                            # 数据集列表 (重构版：全宽)
                            with ui.card().classes('w-full flex-grow border rounded-xl bg-white p-4 shadow-sm no-shadow flex flex-col'):
                                with ui.row().classes('w-full items-center justify-between mb-2 shrink-0'):
                                    with ui.row().classes('items-center gap-4'):
                                        ui.label('📂 数据集列表').classes('text-lg font-bold')
                                        # 新建数据集按钮
                                        def show_create_dialog():
                                            with ui.dialog() as d, ui.card():
                                                ui.label('新建数据集').classes('text-lg font-bold')
                                                name_input = ui.input('数据集名称').classes('w-64').props('autofocus')
                                                def create():
                                                    name = name_input.value.strip()
                                                    if not name:
                                                        ui.notify('名称不能为空', type='warning')
                                                        return
                                                    new_path = os.path.join(DATASET_ROOT, name)
                                                    if os.path.exists(new_path):
                                                        ui.notify('数据集已存在', type='warning')
                                                        return
                                                    try:
                                                        os.makedirs(new_path)
                                                        ui.notify(f'已创建: {name}', type='positive')
                                                        refresh_ds_list()
                                                        refresh_gallery(force=True) # 刷新全局数据集下拉列表
                                                        d.close()
                                                    except Exception as e:
                                                        ui.notify(f'创建失败: {e}', type='negative')
                                                
                                                with ui.row().classes('w-full justify-end mt-4'):
                                                    ui.button('取消', on_click=d.close).props('flat')
                                                    ui.button('创建', on_click=create)
                                            d.open()
                                            
                                        ui.button('新建数据集', on_click=show_create_dialog, icon='add').props('flat dense').classes('bg-blue-50 text-blue-600')

                                    # 刷新按钮 (同时更新顶部数据集下拉列表)
                                    ui.button(icon='refresh', on_click=lambda: [refresh_ds_list(), refresh_gallery(force=True)]).props('flat round dense')

                                # 列表容器 (增加高度)
                                ds_list_scroll = ui.scroll_area().classes('w-full flex-grow border rounded bg-gray-50 p-2 mb-4')
                                
                                # 操作按钮区域
                                with ui.column().classes('w-full gap-2 shrink-0'):
                                    # Row 1: Download & Delete
                                    with ui.row().classes('w-full gap-2'):
                                        ui.button('⬇️ 下载 ZIP', on_click=lambda: download_zip()).classes('flex-1 bg-blue-600 text-white')
                                        ui.button('🗑️ 删除数据集', on_click=lambda: delete_ds()).classes('flex-1 bg-red-600 text-white')
                                    
                                    # Row 2: Uploads
                                    with ui.row().classes('w-full gap-2'):
                                        # Upload Files
                                        # 修复：使用 run_method('pickFiles') 触发文件选择
                                        with ui.button('📤 上传文件', on_click=lambda: upload_files_uploader.run_method('pickFiles')).classes('flex-1 bg-gray-100 text-gray-800 border'):
                                            ui.tooltip('上传图片/视频到当前选中的数据集')
                                        
                                        # Upload ZIP
                                        with ui.button('📦 上传 ZIP', on_click=lambda: upload_zip_uploader.run_method('pickFiles')).classes('flex-1 bg-gray-100 text-gray-800 border'):
                                            ui.tooltip('上传压缩包并自动解压为新数据集')

                                # 隐藏的 Uploaders
                                def get_upload_info(e):
                                    """
                                    Standardized extraction for NiceGUI UploadEventArguments.
                                    """
                                    try:
                                        # 1. Primary: e.file (New NiceGUI API)
                                        # In newer NiceGUI, 'e.file' is the SpooledTemporaryFile and has the 'name' attribute.
                                        if hasattr(e, 'file'):
                                             # e.file is a SpooledTemporaryFile-like object
                                             return e.file.name, e.file
                                        
                                        # 2. Legacy: e.name / e.content
                                        fname = getattr(e, 'name', None) or getattr(e, 'filename', None)
                                        return fname, e.content
                                    except Exception as ex:
                                        print(f"DEBUG: Upload info extraction failed: {ex}")
                                        return None, None

                                async def handle_file_upload(e):
                                    if not state.selected_ds_manage:
                                        ui.notify("请先在列表中选择一个数据集！", type="warning")
                                        upload_files_uploader.reset() # Reset to allow retry
                                        return
                                    
                                    fname, content = get_upload_info(e)
                                    
                                    if not fname:
                                        ui.notify("上传失败: 无法获取文件名", type="negative")
                                        upload_files_uploader.reset()
                                        return
                                    if not content:
                                        ui.notify("上传失败: 无法读取文件内容", type="negative")
                                        upload_files_uploader.reset()
                                        return

                                    target_dir = os.path.join(DATASET_ROOT, state.selected_ds_manage)
                                    try:
                                        save_path = os.path.join(target_dir, fname)
                                        # NiceGUI 2.0+ FileUpload object
                                        if hasattr(content, 'save'):
                                            await content.save(save_path)
                                        else:
                                            # Legacy fallback
                                            if hasattr(content, 'seek'): content.seek(0)
                                            with open(save_path, 'wb') as f:
                                                if hasattr(content, 'read'):
                                                    shutil.copyfileobj(content, f)
                                                else:
                                                    f.write(content)
                                                    
                                        ui.notify(f"已上传: {fname}")
                                        upload_files_uploader.reset() # Clear success files
                                        refresh_ds_list()
                                        
                                        # If uploaded to current viewing folder, refresh gallery
                                        if os.path.abspath(target_dir) == os.path.abspath(state.config["current_folder"]):
                                            refresh_gallery(force=True)
                                            
                                    except Exception as ex:
                                        ui.notify(f"上传失败: {ex}", type="negative")
                                        upload_files_uploader.reset()
                                
                                upload_files_uploader = ui.upload(on_upload=handle_file_upload, multiple=True, auto_upload=True).classes('hidden')

                                async def handle_zip_upload(e):
                                    try:
                                        import zipfile
                                        import tempfile
                                        
                                        filename, content = get_upload_info(e)
                                        
                                        if not filename:
                                            ui.notify("无法获取文件名", type="negative")
                                            upload_zip_uploader.reset()
                                            return
                                        if not content:
                                            ui.notify("无法读取ZIP内容", type="negative")
                                            upload_zip_uploader.reset()
                                            return

                                        # Save zip to temp
                                        tmp_path = os.path.join(tempfile.gettempdir(), filename)
                                        
                                        # NiceGUI 2.0+ FileUpload object
                                        if hasattr(content, 'save'):
                                            await content.save(tmp_path)
                                        else:
                                            # Legacy fallback
                                            if hasattr(content, 'seek'): content.seek(0)
                                            with open(tmp_path, 'wb') as f:
                                                if hasattr(content, 'read'):
                                                    shutil.copyfileobj(content, f)
                                                else:
                                                    f.write(content)
                                        
                                        # Extract strategy: Extract to DATASET_ROOT/Filename_NoExt
                                        folder_name = os.path.splitext(filename)[0]
                                        extract_path = os.path.join(DATASET_ROOT, folder_name)
                                        if not os.path.exists(extract_path):
                                            os.makedirs(extract_path)
                                            
                                        with zipfile.ZipFile(tmp_path, 'r') as z:
                                            z.extractall(extract_path)
                                        
                                        ui.notify(f"已解压到: {folder_name}", type="positive")
                                        upload_zip_uploader.reset()
                                        refresh_ds_list()
                                        
                                        # Auto switch to new dataset
                                        change_dataset(folder_name)
                                        if hasattr(state, 'ds_select'):
                                            state.ds_select.value = folder_name
                                        
                                        try: os.remove(tmp_path)
                                        except: pass
                                    except Exception as ex:
                                        ui.notify(f"解压失败: {ex}", type="negative")
                                        upload_zip_uploader.reset()
                                        print(f"ZIP Upload Error: {ex}")

                                upload_zip_uploader = ui.upload(on_upload=handle_zip_upload, multiple=False, auto_upload=True).props('accept=".zip"').classes('hidden')

                                # 列表渲染逻辑
                                def refresh_ds_list():
                                    ds_list_scroll.clear()
                                    try:
                                        datasets = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
                                        datasets.sort()
                                    except:
                                        datasets = []
                                    
                                    # 更新顶部下拉列表 (通过调用 refresh_gallery)
                                    # 但为了避免循环调用或性能问题，我们在修改操作后显式调用，这里只刷新列表 UI
                                    
                                    with ds_list_scroll:
                                        for ds in datasets:
                                            ds_path = os.path.join(DATASET_ROOT, ds)
                                            try:
                                                stat = os.stat(ds_path)
                                                ctime = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M')
                                                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                                            except:
                                                ctime = "-"
                                                mtime = "-"
                                            
                                            # Highlighting
                                            is_selected = state.selected_ds_manage == ds
                                            bg_class = "bg-blue-100 border-blue-500" if is_selected else "bg-white hover:bg-gray-50 border-transparent"
                                            
                                            with ui.row().classes(f'w-full p-2 border rounded cursor-pointer transition-colors mb-1 items-center justify-between {bg_class}') as row:
                                                # Name
                                                with ui.row().classes('items-center gap-2'):
                                                    ui.icon('folder', color='blue-500' if is_selected else 'gray-400')
                                                    ui.label(ds).classes('font-medium')
                                                
                                                # Metadata
                                                with ui.column().classes('items-end text-xs text-gray-400'):
                                                    ui.label(f"创建: {ctime}")
                                                    ui.label(f"修改: {mtime}")
                                                
                                                # Click handler (use local scope capture)
                                                def on_click(e, d=ds):
                                                    if state.selected_ds_manage == d:
                                                        # Toggle off
                                                        state.selected_ds_manage = None
                                                    else:
                                                        state.selected_ds_manage = d
                                                    refresh_ds_list()
                                                
                                                row.on('click', on_click)
                                
                                # Helper implementations
                                def download_zip():
                                    if not state.selected_ds_manage:
                                        ui.notify("请先选择要下载的数据集", type="warning")
                                        return
                                    try:
                                        ds_path = os.path.join(DATASET_ROOT, state.selected_ds_manage)
                                        import tempfile
                                        tmp_dir = tempfile.gettempdir()
                                        zip_name = f"{state.selected_ds_manage}_{int(time.time())}"
                                        base_path = os.path.join(tmp_dir, zip_name)
                                        
                                        zip_file = shutil.make_archive(base_path, 'zip', ds_path)
                                        ui.download(zip_file, filename=f"{state.selected_ds_manage}.zip")
                                        ui.notify("下载已开始", type="positive")
                                    except Exception as ex:
                                        ui.notify(f"打包失败: {ex}", type="negative")

                                def delete_ds():
                                    if not state.selected_ds_manage:
                                        ui.notify("请先选择要删除的数据集", type="warning")
                                        return
                                    
                                    # 将确认逻辑定义在 dialog 内部以确保作用域清晰
                                    dataset_to_delete = state.selected_ds_manage
                                    
                                    with ui.dialog() as del_dialog, ui.card():
                                        ui.label(f"确定要删除数据集 '{dataset_to_delete}' 吗？").classes('font-bold')
                                        ui.label("此操作无法撤销。").classes('text-sm text-red-500')
                                        
                                        def do_delete_action():
                                            try:
                                                target_path = os.path.join(DATASET_ROOT, dataset_to_delete)
                                                if os.path.exists(target_path):
                                                    shutil.rmtree(target_path)
                                                    ui.notify(f"已删除: {dataset_to_delete}")
                                                else:
                                                    ui.notify("数据集不存在，可能已被删除", type="warning")
                                                    
                                                state.selected_ds_manage = None
                                                # 刷新列表
                                                refresh_ds_list()
                                                # 刷新全局下拉
                                                refresh_gallery(force=True) 
                                                del_dialog.close()
                                            except Exception as ex:
                                                ui.notify(f"删除失败: {ex}", type="negative")

                                        with ui.row().classes('w-full justify-end mt-4'):
                                            ui.button('取消', on_click=del_dialog.close).props('flat')
                                            ui.button('确定删除', on_click=do_delete_action).classes('bg-red text-white')
                                    
                                    del_dialog.open()

                        # Initial load
                        refresh_ds_list()

                    # --- TAB 4: 详细信息 ---
                    with ui.tab_panel(tab_meta).classes('w-full h-full p-6 overflow-y-auto'):
                        ui.label('ℹ️ 详细信息').classes('text-2xl font-bold mb-6')
                        
                        details_container = ui.column().classes('w-full gap-4')

                        def format_size(size_bytes):
                            if size_bytes < 1024: return f"{size_bytes} B"
                            elif size_bytes < 1024*1024: return f"{size_bytes/1024:.2f} KB"
                            else: return f"{size_bytes/(1024*1024):.2f} MB"

                        def get_image_info(path):
                            try:
                                with Image.open(path) as img:
                                    return img.format, img.size
                            except:
                                return "Unknown", (0, 0)

                        def refresh_details_ui():
                            details_container.clear()
                            
                            selected = list(state.selected_files)
                            count = len(selected)
                            
                            with details_container:
                                if count == 0:
                                    ui.label("请先选择图片以查看详细信息").classes('text-gray-500 italic')
                                    return

                                if count == 1:
                                    f_path = selected[0]
                                    if not os.path.exists(f_path):
                                        ui.label("文件不存在").classes('text-red-500')
                                        return
                                        
                                    stat = os.stat(f_path)
                                    ctime = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                                    size_str = format_size(stat.st_size)
                                    fmt, res = get_image_info(f_path)
                                    dataset_name = os.path.basename(os.path.dirname(f_path))
                                    
                                    with ui.card().classes('w-full max-w-2xl p-6 bg-white border shadow-sm'):
                                        ui.label("基本信息").classes('text-lg font-bold mb-4 text-gray-700')
                                        
                                        with ui.grid(columns=2).classes('w-full gap-y-4 gap-x-8'):
                                            ui.label("名称").classes('text-gray-500')
                                            ui.label(os.path.basename(f_path)).classes('font-medium break-all')
                                            
                                            ui.label("格式").classes('text-gray-500')
                                            ui.label(fmt).classes('font-medium')
                                            
                                            ui.label("所属数据集").classes('text-gray-500')
                                            ui.label(dataset_name).classes('font-medium')
                                            
                                            ui.label("创建日期").classes('text-gray-500')
                                            ui.label(ctime).classes('font-medium')
                                            
                                            ui.label("修改日期").classes('text-gray-500')
                                            ui.label(mtime).classes('font-medium')
                                            
                                            ui.label("大小").classes('text-gray-500')
                                            ui.label(size_str).classes('font-medium')
                                            
                                            ui.label("分辨率").classes('text-gray-500')
                                            ui.label(f"{res[0]} x {res[1]}").classes('font-medium')
                                
                                else: # Multiple files
                                    total_size = 0
                                    resolutions = set()
                                    dataset_names = set()
                                    
                                    for f_path in selected:
                                        if os.path.exists(f_path):
                                            total_size += os.path.getsize(f_path)
                                            _, res = get_image_info(f_path)
                                            resolutions.add(f"{res[0]} x {res[1]}")
                                            dataset_names.add(os.path.basename(os.path.dirname(f_path)))
                                    
                                    ds_display = ", ".join(sorted(list(dataset_names)))
                                    if len(dataset_names) > 3:
                                        ds_display = f"{len(dataset_names)} 个数据集"
                                        
                                    res_display = ", ".join(sorted(list(resolutions)))
                                    
                                    with ui.card().classes('w-full max-w-2xl p-6 bg-white border shadow-sm'):
                                        ui.label(f"已选中 {count} 个文件").classes('text-lg font-bold mb-4 text-blue-600')
                                        
                                        with ui.grid(columns=2).classes('w-full gap-y-4 gap-x-8'):
                                            ui.label("所属数据集").classes('text-gray-500')
                                            ui.label(ds_display).classes('font-medium')
                                            
                                            ui.label("数量").classes('text-gray-500')
                                            ui.label(str(count)).classes('font-medium')
                                            
                                            ui.label("总大小").classes('text-gray-500')
                                            ui.label(format_size(total_size)).classes('font-medium')
                                            
                                            ui.label("分辨率规格").classes('text-gray-500')
                                            ui.label(res_display).classes('font-medium')

                        # Register callback
                        ui_callbacks["refresh_details"] = refresh_details_ui
                        
                        # Initial load
                        refresh_details_ui()

                    # --- TAB 5: 设置 ---
                    with ui.tab_panel(tab_settings).classes('w-full h-full p-6 overflow-y-auto'):
                        ui.label('⚙️ 设置').classes('text-2xl font-bold mb-6')
                        
                        with ui.card().classes('w-full max-w-3xl p-6 bg-white shadow-md border mb-6'):
                            ui.label('系统信息').classes('text-lg font-bold mb-4')
                            
                            with ui.grid(columns=2).classes('gap-4 text-gray-600'):
                                ui.label(f"Python Version: {os.sys.version.split()[0]}")
                                ui.label(f"PyTorch: {torch.__version__ if torch else 'Not Installed'}")
                                ui.label(f"CUDA Available: {torch.cuda.is_available() if torch else False}")
                                ui.label(f"GPU Model: {torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else 'N/A'}")
                                ui.label(f"CPU Model: {get_cpu_model()}")
                                
                                # Dynamic Info
                                sys_cpu_label = ui.label('CPU: -')
                                sys_ram_label = ui.label('RAM: -')
                                sys_gpu_label = ui.label('GPU Load: -')
                                sys_vram_label = ui.label('VRAM: -')

                            ui.separator().classes('my-4')
                            ui.label('性能监视器配置').classes('font-bold text-gray-700')
                            ui.switch('在顶部显示实时性能面板 (Performance Monitor in Header)', 
                                    value=state.config.get("show_perf_monitor", False),
                                    on_change=lambda e: update_config("show_perf_monitor", e.value)).props('color=blue')

                            def update_settings_sys_info():
                                try:
                                    s = get_system_stats()
                                    sys_cpu_label.text = f"CPU: {s['cpu']}%"
                                    sys_ram_label.text = f"RAM: {int(s['ram_used'])}/{int(s['ram_total'])} GB ({s['ram_percent']}%)"
                                    sys_gpu_label.text = f"GPU Load: {s['gpu']}%"
                                    sys_vram_label.text = f"VRAM: {s['vram_used']} / {s['vram_total']} MB"
                                except: pass
                            
                            ui.timer(5.0, update_settings_sys_info)

                        with ui.card().classes('w-full max-w-3xl p-6 bg-white shadow-md border'):
                            ui.label('🏷️ 自定义标签管理').classes('text-lg font-bold mb-4')
                            
                            with ui.row().classes('w-full gap-4 items-end mb-4'):
                                new_tag_input = ui.input('添加新标签').classes(INPUT_STYLE + ' w-64')
                                def add_tag_setting():
                                    val = new_tag_input.value.strip()
                                    if val and val not in state.config["custom_quick_tags"]:
                                        state.config["custom_quick_tags"].append(val)
                                        state.save_config()
                                        refresh_tags_ui()
                                        refresh_quick_tags() # 更新侧边栏
                                        new_tag_input.value = ""
                                ui.button('添加', on_click=add_tag_setting).classes(BTN_PRIMARY)

                            tags_container = ui.row().classes('w-full gap-2 flex-wrap p-4 border rounded bg-gray-50')
                            def refresh_tags_ui():
                                tags_container.clear()
                                with tags_container:
                                    if not state.config["custom_quick_tags"]:
                                        ui.label("暂无自定义标签").classes("text-gray-400")
                                    for t in state.config["custom_quick_tags"]:
                                        def remove_tag_wrapper(tag_to_remove=t):
                                            if tag_to_remove in state.config["custom_quick_tags"]:
                                                state.config["custom_quick_tags"].remove(tag_to_remove)
                                                state.save_config()
                                                refresh_tags_ui()
                                                refresh_quick_tags()
                                        
                                        with ui.row().classes('items-center bg-white border rounded px-3 py-1 gap-2 shadow-sm'):
                                            ui.label(t).classes('text-sm font-medium')
                                            ui.icon('close', size='xs').classes('cursor-pointer text-red-500 hover:text-red-700').on('click', remove_tag_wrapper)
                            
                            ui_callbacks["refresh_tags_ui"] = refresh_tags_ui
                            refresh_tags_ui()

    # --- 逻辑绑定 ---
    def update_config(key, value):
        state.config[key] = value
        state.save_config()
        if key == "source_type": render_ai_config()

    def change_dataset(name):
        new_path = os.path.join(DATASET_ROOT, name)
        state.config["current_folder"] = new_path
        state.save_config()
        refresh_gallery()

    def stop_worker():
        if worker and worker.is_alive():
            worker.should_stop = True
            ui.notify("正在停止...")

    def update_ui_loop():
        if state.is_processing:
            progress_bar.value = state.process_progress
            status_label.text = state.process_status
        
        # log_area 已被移除，此处代码应删除或注释
        # log_area.clear()
        # with log_area:
        #    for log in reversed(state.logs):
        #        ui.label(log)



    # ui.timer(0.5, update_ui_loop) # Removed duplicate timer to save resources
    ui.timer(0.1, refresh_gallery, once=True)

ui.run(title="DocCaptioner v1.0", host="127.0.0.1", port=9090, reload=True, dark=False, storage_secret="secret")