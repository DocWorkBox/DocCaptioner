import os
import time
import gc
import threading
import base64
import requests
import numpy as np
import cv2
import warnings
from io import BytesIO
from PIL import Image

# 忽略 bitsandbytes 的 MatMul8bitLt 警告 (已知无害且无法消除)
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
warnings.filterwarnings("ignore", module="bitsandbytes.autograd._functions")

# Third-party imports
try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    from huggingface_hub import snapshot_download
except ImportError:
    torch = None
    BitsAndBytesConfig = None

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

from app.config import state, KNOWN_MODELS
from app.utils import is_video, save_caption

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
                                    snapshot_download(repo_id=repo_id, local_dir=std_dir, resume_download=True)
                                    model_path = std_dir
                                except Exception as e:
                                    raise RuntimeError(f"模型下载失败: {e}")
                    else:
                        model_path = self.config.get("local_model_path", "")

                    if not model_path or not os.path.exists(model_path):
                         raise RuntimeError("无效的模型路径")

                    if self.config.get("selected_model_key", "").endswith("GGUF") or model_path.endswith(".gguf"):
                        if not HAS_GGUF: raise RuntimeError("未安装 llama-cpp-python")
                        
                        chat_handler = Llava15ChatHandler(clip_model_path=os.path.dirname(model_path))
                        # Offload layers to GPU as much as possible, but respect limit
                        model = Llama(model_path=model_path, chat_handler=chat_handler, n_ctx=2048, n_gpu_layers=-1)
                    else:
                        if torch is None: raise RuntimeError("未安装 PyTorch/Transformers")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        # --- 显存/性能优化配置 ---
                        load_kwargs = {
                            "trust_remote_code": True,
                            # 修复 torch_dtype 警告，使用 dtype
                            "dtype": torch.float16 if device=="cuda" else torch.float32, 
                        }
                        
                        # 提前获取 quant_mode 以供显存计算使用
                        quant_mode = self.config.get("quantization", "None")

                        if device == "cuda":
                            # 显存管理策略
                            use_vram_opt = self.config.get("vram_optimization", False)
                            
                            if quant_mode == "None" and use_vram_opt:
                                # 只有在【非量化模式】且【用户开启显存优化】时，才启用 CPU Offload
                                try:
                                    vram_total = torch.cuda.get_device_properties(0).total_memory
                                    reserve_bytes = max(4 * 1024**3, int(vram_total * 0.3)) # 预留 30% 或 4GB
                                    limit_bytes = vram_total - reserve_bytes
                                    limit_gib = limit_bytes / (1024**3)
                                    
                                    load_kwargs["device_map"] = "auto"
                                    load_kwargs["max_memory"] = {0: limit_bytes, "cpu": "256GiB"}
                                    state.add_log(f"显存优化: 限制使用 {limit_gib:.2f}GiB (预留 {reserve_bytes/(1024**3):.2f}GiB), 剩余部分 Offload 到 CPU")
                                except Exception as e:
                                    state.add_log(f"显存优化设置失败: {e}，将使用全量模式")
                                    load_kwargs["device_map"] = "cuda"
                            else:
                                # 量化模式或用户未开启优化 -> 全量 GPU 模式
                                load_kwargs["device_map"] = "cuda"
                                if quant_mode != "None":
                                     state.add_log(f"显存优化: 量化模式下已自动禁用 (优先兼容性)")
                                else:
                                     state.add_log(f"显存优化: 未启用 (全量 GPU 模式)")
                        else:
                            load_kwargs["device_map"] = "cpu"

                        # 1. Quantization (4-bit / 8-bit)
                        # quant_mode = self.config.get("quantization", "None") # 已在上方定义
                        if device == "cuda" and quant_mode != "None" and BitsAndBytesConfig:
                            # 检查 bitsandbytes 是否可用
                            try:
                                import bitsandbytes as bnb
                                if quant_mode == "4-bit":
                                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                    )
                                    state.add_log("启用 4-bit 量化 (NF4)")
                                elif quant_mode == "8-bit":
                                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                        load_in_8bit=True,
                                    )
                                    state.add_log("启用 8-bit 量化")
                            except ImportError:
                                state.add_log("⚠️ Warning: bitsandbytes 未安装，量化功能已禁用，将使用默认精度。")
                            except Exception as e:
                                state.add_log(f"⚠️ Warning: bitsandbytes 加载失败: {e}，将使用默认精度。")

                        # 2. Flash Attention 2 (Ampere+ GPUs)
                        if device == "cuda":
                            try:
                                # 检查是否安装了 flash_attn
                                import importlib.util
                                has_fa2 = importlib.util.find_spec("flash_attn") is not None
                                
                                major, _ = torch.cuda.get_device_capability(0)
                                if major >= 8 and has_fa2:
                                    load_kwargs["attn_implementation"] = "flash_attention_2"
                                    state.add_log("检测到 Ampere+ 显卡且已安装 flash_attn，启用 Flash Attention 2 加速")
                                elif major >= 8:
                                    # 有显卡但没包，静默回退或提示
                                    # load_kwargs["attn_implementation"] = "sdpa" # PyTorch 2.0+ default
                                    state.add_log("使用 PyTorch 内置 SDPA 加速 (性能接近 Flash Attention)")
                            except: pass

                        # VRAM Safety Check
                        if device == "cuda":
                            vram_total = torch.cuda.get_device_properties(0).total_memory
                            vram_gb = vram_total / (1024**3)
                            state.add_log(f"Detected VRAM: {vram_gb:.1f} GB")
                            
                            if vram_gb < 12 and "8B" in model_path and quant_mode == "None":
                                state.add_log("⚠️ 警告: 显存可能不足 (<12GB)，建议开启 4-bit 量化或使用 GGUF 版本")

                        try:
                            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                            model = AutoModelForImageTextToText.from_pretrained(
                                model_path, **load_kwargs
                            ).eval()
                                
                        except Exception as e:
                            err_str = str(e)

                            if "no kernel image is available" in err_str:
                                raise RuntimeError("您的显卡 (RTX 50系列/其他) 与当前 PyTorch 版本不兼容。请使用 API 模式。")
                            
                            # 自动降级重试逻辑
                            is_bnb_error = "bitsandbytes" in err_str.lower()
                            is_fa2_error = "flash_attn" in err_str.lower() or "flash attention" in err_str.lower()
                            is_model_error = "no file named" in err_str or "safetensor" in err_str.lower() or "oserror" in err_str.lower()
                            
                            if is_bnb_error or is_fa2_error:
                                state.add_log(f"⚠️ 优化组件加载失败 ({'BitsAndBytes' if is_bnb_error else 'FlashAttn'}), 正在尝试降级重试...")
                                
                                if "quantization_config" in load_kwargs:
                                    del load_kwargs["quantization_config"]
                                    state.add_log("- 已禁用量化")
                                
                                if "attn_implementation" in load_kwargs:
                                    del load_kwargs["attn_implementation"]
                                    state.add_log("- 已禁用 Flash Attention")
                                
                                # 重试加载
                                model = AutoModelForImageTextToText.from_pretrained(
                                    model_path, **load_kwargs
                                ).eval()
                                state.add_log("✅ 降级模式加载成功")
                            
                            # 断点续传逻辑
                            elif is_model_error and source_type == "预设模型 (Preset)":
                                state.add_log(f"⚠️ 检测到模型文件可能损坏或缺失: {e}")
                                state.add_log("⏳ 正在尝试断点续传/修复模型...")
                                
                                try:
                                    # 重新触发下载
                                    model_info = KNOWN_MODELS.get(self.config["selected_model_key"])
                                    repo_id = model_info["repo_id"]
                                    snapshot_download(repo_id=repo_id, local_dir=model_path, resume_download=True)
                                    state.add_log("✅ 模型修复完成，正在重新加载...")
                                    
                                    # 再次重试加载
                                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                                    model = AutoModelForImageTextToText.from_pretrained(
                                        model_path, **load_kwargs
                                    ).eval()
                                    state.add_log("✅ 模型重新加载成功")
                                except Exception as download_err:
                                    raise RuntimeError(f"模型修复失败: {download_err}")
                            else:
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
                        
                        is_dashscope = "qwen" in self.config["api_model_name"].lower() and "plus" in self.config["api_model_name"].lower() and "vl" in self.config["api_model_name"].lower()
                        
                        # 构造内容
                        if is_dashscope and "dashscope" in api_base:
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
                            
                            # 显式转换浮点类型输入到 float16
                            if device == "cuda":
                                for k, v in inputs.items():
                                    if torch.is_tensor(v) and torch.is_floating_point(v) and v.dtype == torch.float32:
                                        inputs[k] = v.to(torch.float16)

                            state.add_log(f"开始推理: {fname}...")
                            
                            with torch.no_grad():
                                # Use autocast for mixed precision
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
                if (i + 1) % 5 == 0 or is_vid: # 每 5 张图或遇到视频时清理一次
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                time.sleep(0.1)

        except Exception as e:
            state.add_log(f"任务异常: {e}")
            print(f"Worker Exception: {e}")
        finally:
            state.is_processing = False
            state.process_status = "任务完成" if not self.should_stop else "任务已停止"
            if not self.should_stop:
                state.process_progress = 1.0
                
            # 刷新画廊逻辑需在 UI 层处理，这里仅更新状态
            
            # 尝试卸载模型
            if self.config.get("unload_model", False) and source_type != "在线 API (OpenAI Compatible)":
                del model
                del processor
                if torch: torch.cuda.empty_cache()
                state.add_log("模型已卸载")
