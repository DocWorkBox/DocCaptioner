import os
import json
import time
from nicegui import ui

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
    },
    "Huihui-Qwen3-VL-8B-Instruct (Abliterated)": {"repo_id": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"}
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
# 使用脚本所在目录作为基准，而不是当前工作目录，确保移动程序后路径正确
# 注意：这里假设 web_app_ng.py 在根目录，而 app/config.py 在 app/ 目录
# 所以我们需要向上走一级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "Dataset Collections")
THUMB_DIR = os.path.join(BASE_DIR, "thumbnails")

if not os.path.exists(DATASET_ROOT):
    os.makedirs(DATASET_ROOT)

if not os.path.exists(THUMB_DIR):
    os.makedirs(THUMB_DIR)

class AppState:
    def __init__(self):
        self.config = self.load_config()
        self.config.setdefault("current_folder", os.path.join(DATASET_ROOT, "default_dataset") if os.path.exists(os.path.join(DATASET_ROOT, "default_dataset")) else DATASET_ROOT)
        self.config.setdefault("source_type", "预设模型 (Preset)")
        self.config.setdefault("selected_model_key", list(KNOWN_MODELS.keys())[0])
        self.config.setdefault("api_model_name", "Qwen/Qwen3-vl-Plus")
        self.config.setdefault("api_base_url", "https://api.openai.com/v1")
        self.config.setdefault("api_key", "")
        self.config.setdefault("prompt_template", "详细描述 (Detailed Description)")
        self.config.setdefault("custom_prompt", "")
        self.config.setdefault("max_tokens", 512)
        self.config.setdefault("temperature", 0.7)
        self.config.setdefault("top_p", 0.9)
        self.config.setdefault("unload_model", False)
        self.config.setdefault("quantization", "None") # Options: None, 4-bit, 8-bit
        self.config.setdefault("custom_quick_tags", [])
        self.config.setdefault("splitter_value", 40)
        self.config.setdefault("show_perf_monitor", False)
        
        self.current_files = []
        self.selected_files = set()
        self.card_refs = {}  # Store references to caption textareas {file_path: textarea_element}
        self.file_timestamps = {} # Store timestamps for images to force refresh
        self.logs = []
        self.is_processing = False
        self.process_progress = 0.0
        self.process_status = "就绪"

        # Model state
        self.current_model = None
        self.processor = None
        self.model_name = None
        self.chat_handler = None # For GGUF

    def load_config(self):
        if os.path.exists(os.path.join(BASE_DIR, CONFIG_FILE)):
            try:
                with open(os.path.join(BASE_DIR, CONFIG_FILE), "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 自动修复数据集路径
                current = config.get("current_folder")
                if current and not os.path.exists(current):
                    folder_name = os.path.basename(current)
                    new_path = os.path.join(DATASET_ROOT, folder_name)
                    if os.path.exists(new_path):
                        print(f"Auto-fixing dataset path: {current} -> {new_path}")
                        config["current_folder"] = new_path
                    elif os.path.exists(DATASET_ROOT):
                        print(f"Dataset path not found: {current}, resetting to root.")
                        config["current_folder"] = DATASET_ROOT
                        
                return config
            except:
                pass
        return {}

    def save_config(self):
        try:
            with open(os.path.join(BASE_DIR, CONFIG_FILE), "w", encoding="utf-8") as f:
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
        
        print(line) # Debug to console

# Create a global state instance
state = AppState()
