import os
import time
import json
import asyncio
import shutil
import zipfile
import requests
import warnings
import piexif
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from nicegui import ui, app
from app.config import state, PROMPTS, KNOWN_MODELS, DATASET_ROOT, BASE_QUICK_TAGS, THUMB_DIR
from app.utils import get_caption, save_caption, toggle_tag, is_video, get_caption_path
from app.system import get_cpu_model, get_system_stats
from app.model import AIWorker
from ui.styles import GLOBAL_CSS, BTN_PRIMARY, BTN_SECONDARY, BTN_DANGER, INPUT_STYLE, CARD_STYLE

try:
    import torch
except ImportError:
    torch = None

worker = None

# Renamed to create_ui to be explicit, but keeping main_page logic
def create_ui():
    ui.page_title('DocCaptioner v1.1')
    
    # 注入全局样式 (包含移动端响应式 + Apple UI 风格)
    ui.add_head_html(GLOBAL_CSS)
    
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
        # 更新当前激活文件并刷新详细信息
        nonlocal current_active_file
        current_active_file = f_path
        ui_callbacks["refresh_details"]()

        # 使用 flex 居中布局，避免强制拉伸
        with ui.dialog() as d, ui.card().classes('w-full h-full max-w-none p-0 bg-black flex items-center justify-center relative'):
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
                # 使用 w-full h-full 确保占满容器，配合 fit=contain (Quasar prop) 和 object-fit: contain (CSS)
                # 这样可以确保图片不被截断，且保持比例，多余部分显示背景色
                ui.image(url_path).classes('w-full h-full').props('fit=contain').style('object-fit: contain;')
            
            # Caption Overlay (Bottom) - 在移动端方便查看打标内容
            caption = get_caption(f_path)
            if caption:
                with ui.column().classes('absolute bottom-0 left-0 w-full bg-black/60 p-4 max-h-[30vh] overflow-y-auto z-40'):
                    ui.label('📝 当前打标:').classes('text-xs text-gray-300 font-bold mb-1')
                    ui.label(caption).classes('text-sm text-white leading-relaxed whitespace-pre-wrap')
        d.open()

    def create_image_card(f_path):
        is_sel = f_path in state.selected_files
        border_class = "border-2 border-blue-500" if is_sel else "border border-gray-200"
        
        # 优化：复用 timestamp
        import time
        ts = int(time.time() * 1000)
        
        if f_path not in state.file_timestamps:
            state.file_timestamps[f_path] = ts
        
        current_ts = state.file_timestamps[f_path]
        
        # 构建相对 URL (用于视频等，或者点击大图)
        rel_path = os.path.relpath(f_path, DATASET_ROOT)
        rel_path = rel_path.replace('\\', '/')
        from urllib.parse import quote
        url_path = f'/datasets/{quote(rel_path)}?t={current_ts}'
        
        # 构建缩略图 URL (用于画廊显示)
        thumb_url = f'/api/thumbnail?path={quote(f_path)}'
        
        # 优化移动端卡片宽度：PC w-64, 移动端 w-full (由父容器 grid 控制)
        # 增加 group 类以便 hover
        with ui.card().classes(f'w-full md:w-64 flex flex-col p-0 gap-0 relative group bg-white border shadow-sm transition-all hover:shadow-md {border_class}') as card:
            # 存储 card 引用 (虽然这里还没有 checkbox 引用，稍后补充)
            # 注意：旧逻辑是 state.card_refs[f_path] = {"txt": txt, "chk": chk, "card": card}
            # 这里我们先不赋值，等构建完内部元素再赋值
            
            # Selection overlay (checkbox)
            # 移动端：保持显示，但稍微缩小
            def on_chk_change(e):
                 if e.value: 
                     state.selected_files.add(f_path)
                     card.classes(remove="border border-gray-200", add="border-2 border-blue-500")
                 else: 
                     state.selected_files.discard(f_path)
                     card.classes(remove="border-2 border-blue-500", add="border border-gray-200")
                 
                 count = len(state.selected_files)
                 selected_count_label.set_text(f"已选 {count}")
                 ui_callbacks["refresh_details"]()

            chk = ui.checkbox(value=is_sel, on_change=on_chk_change).classes('absolute top-0 right-0 z-10 bg-white/80 rounded-bl-lg transform scale-75 md:scale-100 m-1')
            
            # Image Area
            # 移动端高度减小 h-20 (80px), PC h-48 (192px)
            # 修复：改用 fit=contain (Quasar props) 防止裁切
            # 添加 bg-gray-100 填充背景
            img = ui.image(thumb_url).props('fit=contain').classes('w-full h-20 md:h-48 bg-gray-100 cursor-pointer select-none').on('click', lambda: show_full_image(f_path))
            
            # Info Area
            # 移动端：大幅简化，只显示文件名（截断），隐藏描述和标签
            with ui.column().classes('w-full p-1 md:p-3 gap-0 md:gap-1'):
                # Filename
                ui.label(os.path.basename(f_path)).classes('text-[10px] md:text-sm font-bold truncate w-full leading-tight')
                
                # Caption Editable Area
                caption = get_caption(f_path)
                # 使用 textarea 恢复编辑功能
                # autogrow: 自动高度
                # debounce=1000: 防抖，停止输入1秒后保存，减少IO
                txt = ui.textarea(value=caption, on_change=lambda e, f=f_path: save_caption(f, e.value)) \
                    .props('borderless dense autogrow rows=1 debounce=1000') \
                    .classes('w-full text-[8px] md:text-xs bg-gray-50 rounded px-1 leading-tight min-h-[1.5em] max-h-[4em] md:max-h-[6em] overflow-hidden')

                # Tags (PC only - 移动端隐藏以节省空间)
                tags = [t.strip() for t in caption.split(',') if t.strip()]
                if tags:
                    with ui.row().classes('gap-1 flex-wrap mt-1 hidden md:flex'):
                        for t in tags[:3]:
                            ui.label(t).classes('bg-blue-100 text-blue-800 text-[10px] px-1 rounded')
                        if len(tags) > 3:
                            ui.label(f"+{len(tags)-3}").classes('text-[10px] text-gray-400')
            
            # 存储完整引用以供 toggle_all 和 refresh_captions 使用
            state.card_refs[f_path] = {"chk": chk, "card": card, "txt": txt}

            # Tooltip
            with ui.tooltip(f"{os.path.basename(f_path)}\n{caption}").classes('bg-gray-800 text-white text-xs'):
                pass 
 

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
                # Cleanup thumbnail
                try:
                    if os.path.exists(f):
                        mtime = os.path.getmtime(f)
                        hash_str = f"{f}_{mtime}"
                        thumb_name = hashlib.md5(hash_str.encode('utf-8')).hexdigest() + ".jpg"
                        thumb_path = os.path.join(THUMB_DIR, thumb_name)
                        if os.path.exists(thumb_path):
                            os.remove(thumb_path)
                except: pass

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
                
                elif op == "清除打标 (Clear Tags)":
                    # 清除打标逻辑：删除对应的 .txt 文件
                    txt_path = get_caption_path(f)
                    if os.path.exists(txt_path):
                        try:
                            os.remove(txt_path)
                            count += 1
                        except Exception as e:
                            print(f"删除失败 {txt_path}: {e}")
                    else:
                        # 如果没有打标文件，也算处理成功（本来就没有）
                        pass
                    # 不需要处理图片，直接跳过后续的图片处理逻辑
                    continue
                            
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
    # 移动端优化：自动高度，允许换行，增加内边距
    with ui.header().classes('bg-white text-gray-800 border-b border-gray-200 h-auto min-h-[3.5rem] py-1 px-2 md:px-4 flex flex-wrap items-center justify-between shadow-sm z-50 gap-y-1'):
        # Left: Title (Order 1)
        with ui.row().classes('items-center gap-1 md:gap-2 shrink-0 order-1'):
            ui.button(icon='menu', on_click=lambda: left_drawer.toggle()).classes('text-gray-600')
            ui.icon('local_offer', size='md', color='blue-600').classes('hidden md:block')
            with ui.column().classes('gap-0'):
                ui.label('DocCaptioner').classes('text-sm md:text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 leading-none')
                ui.label('v1.1').classes('text-[8px] md:text-[10px] text-gray-400 leading-none')

        # Selector & Actions (Order 2 on Mobile, Order 3 on Desktop)
        # 移动端：放在 Title 右侧
        with ui.row().classes('items-center gap-1 order-2 md:order-3 ml-auto md:ml-0'):
            # 数据集选择器美化
            # 移动端：缩小选择器宽度 w-14
            with ui.row().classes('items-center gap-1 bg-gray-100 rounded-lg px-1 md:px-2 py-1 border border-gray-200 shrink-0'):
                ui.icon('folder_open', color='gray-500').classes('text-[10px] md:text-sm')
                # 将 ds_select 存储在 state 中以便全局访问
                # 移动端优化：缩小宽度 w-14 (56px), PC w-40 (160px)
                # 缩小字体 text-[10px]
                state.ds_select = ui.select(
                    options=[d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))],
                    value=os.path.basename(state.config["current_folder"]),
                    on_change=lambda e: change_dataset(e.value)
                ).classes('w-14 md:w-40 text-[10px] md:text-sm').props('dense borderless options-dense behavior="menu"') # borderless 融入背景
            
            ui.button(icon='refresh', on_click=lambda: refresh_gallery(force=True)).props('flat round dense').classes('text-gray-600 hover:bg-gray-100 shrink-0')

        # Perf Monitor (Order 3 on Mobile, Order 2 on Desktop)
        # 移动端：独占一行 (w-full)，居中显示
        with ui.row().classes('items-center gap-1 md:gap-3 border-gray-300 order-3 md:order-2 w-full md:w-auto justify-center md:justify-end md:flex-1 md:border-r md:pr-4').bind_visibility_from(state.config, 'show_perf_monitor'):
             # CPU
            with ui.row().classes('items-center gap-0.5 md:gap-1 flex-nowrap'):
                ui.label('CPU').classes('text-[8px] md:text-xs font-bold text-black whitespace-nowrap')
                with ui.element('div').classes('relative w-8 md:w-16 h-3 md:h-4 bg-gray-200 rounded overflow-hidden shrink-0'): # 稍微放大宽度 w-6->w-8, h-2->h-3
                    cpu_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full')
                    cpu_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[7px] md:text-[10px] font-bold text-black z-10')

            # RAM
            with ui.row().classes('items-center gap-0.5 md:gap-1 flex-nowrap'):
                ui.label('RAM').classes('text-[8px] md:text-xs font-bold text-black whitespace-nowrap')
                with ui.element('div').classes('relative w-8 md:w-16 h-3 md:h-4 bg-gray-200 rounded overflow-hidden shrink-0'):
                    ram_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full text-purple-500').props('color=purple')
                    ram_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[7px] md:text-[10px] font-bold text-black z-10')

            # GPU
            with ui.row().classes('items-center gap-0.5 md:gap-1 flex-nowrap'):
                ui.label('GPU').classes('text-[8px] md:text-xs font-bold text-black whitespace-nowrap')
                with ui.element('div').classes('relative w-8 md:w-16 h-3 md:h-4 bg-gray-200 rounded overflow-hidden shrink-0'):
                    gpu_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full text-green-500').props('color=green')
                    gpu_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[7px] md:text-[10px] font-bold text-black z-10')

            # VRAM
            with ui.row().classes('items-center gap-0.5 md:gap-1 flex-nowrap'):
                ui.label('VRAM').classes('text-[8px] md:text-xs font-bold text-black whitespace-nowrap')
                with ui.element('div').classes('relative w-8 md:w-16 h-3 md:h-4 bg-gray-200 rounded overflow-hidden shrink-0'):
                    vram_bar = ui.linear_progress(value=0, show_value=False).classes('absolute inset-0 h-full text-orange-500').props('color=orange')
                    vram_label = ui.label('0%').classes('absolute inset-0 flex items-center justify-center text-[7px] md:text-[10px] font-bold text-black z-10')
            
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
    # responsive-splitter: 自定义 CSS 类，用于在移动端强制垂直布局
    with ui.splitter(value=state.config["splitter_value"], limits=(20, 80), 
                     on_change=lambda e: update_config("splitter_value", e.value)).classes('w-full h-[calc(100vh-5rem)] responsive-splitter') as splitter:
        
        # --- 左侧：画廊列表 (始终显示) ---
        with splitter.before:
            # 增加 md:rounded-l-lg 等类名优化 PC 端圆角，移动端则不需要
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
                # 移动端优化：使用 grid 布局实现一行四列，gap 极小，w-full
                # PC端：保持 flex wrap justify-center, w-64
                gallery_container = ui.element('div').classes('w-full grid grid-cols-4 gap-1 p-1 md:flex md:flex-wrap md:gap-3 md:justify-center md:p-4')
                
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
                # 移动端优化：Tabs 样式
                # flex-wrap: 允许换行
                # justify-center: 居中
                # text-[10px]: 缩小字体
                with ui.tabs(value='tab_ai', on_change=lambda e: None).classes('w-full border-b border-gray-200 bg-white gap-1 md:gap-2 justify-between md:justify-start px-2 shrink-0') as tabs:
                    # 移动端：图标在上，文字在下？或者只显示图标？
                    # 暂时保持图标+文字，但缩小字体和内边距
                    # 使用 flex-1 让它们平分宽度
                    tab_ai = ui.tab('tab_ai', label='AI 自动打标', icon='smart_toy').classes('flex-1 text-[10px] md:text-sm px-1 min-h-[3rem]')
                    tab_batch = ui.tab('批量处理', icon='photo_library').classes('flex-1 text-[10px] md:text-sm px-1 min-h-[3rem]')
                    tab_dataset = ui.tab('数据集', icon='folder_shared').classes('flex-1 text-[10px] md:text-sm px-1 min-h-[3rem]')
                    tab_meta = ui.tab('详细信息', icon='info').classes('flex-1 text-[10px] md:text-sm px-1 min-h-[3rem]')
                    tab_settings = ui.tab('设置', icon='settings').classes('flex-1 text-[10px] md:text-sm px-1 min-h-[3rem]')

                with ui.tab_panels(tabs, value=tab_ai).classes('w-full h-full flex-grow p-0 bg-white min-h-0 overflow-hidden'):
                    
                    # --- TAB 1: AI 标注 ---
                    # 移动端：h-auto (自然高度), overflow-visible (不剪裁，由页面滚动)
                    # PC端：h-full (填满右侧面板), overflow-y-auto (内部滚动)
                    # --- TAB 1: AI 自动打标 (Fixed Layout) ---
                    with ui.tab_panel(tab_ai).classes('w-full h-full p-0 flex flex-col overflow-hidden'):
                        # Scrollable Content
                        with ui.column().classes('w-full flex-grow overflow-y-auto p-4 md:p-6 gap-4 min-h-0 pb-32'):
                            ui.label('🤖 AI 自动打标').classes('text-xl md:text-2xl font-bold mb-2')
                        
                            # --- 模型配置卡片 ---
                            with ui.card().classes(CARD_STYLE + ' w-full'):
                                ui.label('模型配置').classes('font-bold text-gray-700 mb-2')
                            
                                # Row 1: Source + Model Select/Path/URL (Top Line)
                                with ui.column().classes('w-full gap-4'):
                                    ui.select(["预设模型 (Preset)", "在线 API (OpenAI Compatible)", "本地路径 (Local Path)"], 
                                              value=state.config["source_type"], label="来源",
                                              on_change=lambda e: update_config("source_type", e.value)).classes('w-full')
                                
                                    # 动态配置区域 1 (Model Select / API URL)
                                    config_area_1 = ui.column().classes('w-full gap-4')
                            
                                # Row 2: Options / API Key (Bottom Line)
                                config_area_2 = ui.column().classes('w-full gap-4 mt-2')

                                def render_ai_config():
                                    config_area_1.clear()
                                    config_area_2.clear()
                                
                                    source = state.config.get("source_type", "预设模型 (Preset)")
                                
                                    if source == "预设模型 (Preset)":
                                        # Row 1 Right: Model Select
                                        with config_area_1:
                                            ui.select(list(KNOWN_MODELS.keys()), value=state.config["selected_model_key"],
                                                    label="选择模型", on_change=lambda e: update_config("selected_model_key", e.value)).classes('w-full')
                                    
                                        # Row 2 Full: Options
                                        with config_area_2:
                                            # VRAM Optimization Checkbox
                                            vram_opt_chk = ui.checkbox("显存优化 (CPU Offload)", value=state.config.get("vram_optimization", False),
                                                on_change=lambda e: update_config("vram_optimization", e.value))
                                        
                                            # Quantization Setting
                                            q_select = ui.select(["None", "4-bit", "8-bit"], value=state.config.get("quantization", "None"),
                                                    label="量化 (Quantization)", 
                                                    on_change=lambda e: update_config("quantization", e.value)).classes('w-full').props('dense options-dense')
                                        
                                            # Unload Checkbox
                                            ui.checkbox("完成后卸载", value=state.config["unload_model"],
                                                    on_change=lambda e: update_config("unload_model", e.value))

                                            # 绑定可见性逻辑
                                            def update_vram_opt_visibility(e=None):
                                                val = q_select.value if e is None else e.value
                                                if val == "None":
                                                    vram_opt_chk.classes(remove="hidden invisible")
                                                else:
                                                    vram_opt_chk.classes(add="invisible") 
                                                    vram_opt_chk.classes(remove="hidden")
                                        
                                            q_select.on_value_change(update_vram_opt_visibility)
                                            update_vram_opt_visibility()

                                    elif source == "本地路径 (Local Path)":
                                        # Row 1 Right: Model Path Input
                                        with config_area_1:
                                            ui.input("模型路径 (文件夹或 .gguf)", value=state.config.get("local_model_path", ""),
                                                    on_change=lambda e: update_config("local_model_path", e.value)).classes('w-full')
                                    
                                        # Row 2 Full: Options
                                        with config_area_2:
                                            vram_opt_chk = ui.checkbox("显存优化 (CPU Offload)", value=state.config.get("vram_optimization", False),
                                                on_change=lambda e: update_config("vram_optimization", e.value))
                                        
                                            q_select = ui.select(["None", "4-bit", "8-bit"], value=state.config.get("quantization", "None"),
                                                    label="量化 (Quantization)", 
                                                    on_change=lambda e: update_config("quantization", e.value)).classes('w-full').props('dense options-dense')
                                        
                                            ui.checkbox("完成后卸载", value=state.config["unload_model"],
                                                    on_change=lambda e: update_config("unload_model", e.value))

                                            # 绑定可见性逻辑
                                            def update_vram_opt_visibility(e=None):
                                                val = q_select.value if e is None else e.value
                                                if val == "None":
                                                    vram_opt_chk.classes(remove="hidden invisible")
                                                else:
                                                    vram_opt_chk.classes(add="invisible")
                                                    vram_opt_chk.classes(remove="hidden")

                                            q_select.on_value_change(update_vram_opt_visibility)
                                            update_vram_opt_visibility()

                                    else: # API
                                        # Row 1 Right: API Base URL
                                        with config_area_1:
                                            ui.input("API Base URL", value=state.config["api_base_url"],
                                                    on_change=lambda e: update_config("api_base_url", e.value)).classes('w-full')
                                    
                                        # Row 2 Full: API Key + Model + Test
                                        with config_area_2:
                                            ui.input("API Key", password=True, value=state.config["api_key"],
                                                    on_change=lambda e: update_config("api_key", e.value)).classes('w-full')
                                            ui.input("Model Name", value=state.config["api_model_name"],
                                                    on_change=lambda e: update_config("api_model_name", e.value)).classes('w-full')
                                        
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

                            # --- 提示词配置卡片 ---
                            with ui.card().classes(CARD_STYLE + ' w-full'):
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

                                with ui.column().classes('w-full gap-4'):
                                    ui.select(list(PROMPTS.keys()), value=state.config["prompt_template"], label="模板",
                                              on_change=lambda e: [update_config("prompt_template", e.value), update_prompt_text()]).classes('w-full')
                                
                                    ui.select(["英语 (English)", "中文 (Chinese)", "中英双语 (Bilingual)"], 
                                              value=state.config.get("target_lang", "英语 (English)"), label="目标语言",
                                              on_change=lambda e: [update_config("target_lang", e.value), update_prompt_text()]).classes('w-full')

                                prompt_input = ui.textarea(value=PROMPTS.get(state.config["prompt_template"], "")).classes('w-full h-32 bg-white')
                            
                                # User Extra Prompt
                                ui.label('用户额外提示词 (User Input / Context)').classes('text-sm text-gray-500 mt-2')
                                user_extra_prompt = ui.textarea(placeholder="输入具体要求、关键词，如：图中出现的人物以D0c来指代，不具体描述外貌细节").props('rows=2').classes('w-full bg-white')
                            
                                # Initialize
                                update_prompt_text()

                            # 进度与控制

                        # Actions Footer (Fixed at Bottom)
                        # Actions Logic (Extracted)
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



                        

            

                        # Fixed Footer (Progress + Buttons)
                    # --- TAB 2: 批量编辑 ---
                    with ui.tab_panel(tab_batch).classes('w-full h-full p-6 overflow-y-auto'):
                        ui.label('✏️ 批量图片处理').classes('text-2xl font-bold mb-6')
                        
                        # 参数面板
                        with ui.card().classes(CARD_STYLE + ' w-full'):
                            ui.label(f"已选择 {len(state.selected_files)} 张图片").classes('mb-4 text-blue-600 font-medium')
                            
                            ui.label('选择操作').classes('text-sm text-gray-500 mb-1')
                            operation = ui.select(
                                options=["转换格式 (Convert)", "调整大小 (Resize)", "旋转 (Rotate)", "裁剪 (Crop)", "顺序重命名 (Rename)", "清除打标 (Clear Tags)", "删除图片 (Delete Images)"],
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
                                # 清除打标不需要额外参数，所以没有对应的 set_visibility
                            
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
                                op = operation.value
                                
                                # 删除图片逻辑
                                if op == "删除图片 (Delete Images)":
                                    target_files = list(state.selected_files)
                                    if not target_files:
                                        ui.notify("请先选择要删除的图片", type="warning")
                                        return

                                    with ui.dialog() as d, ui.card():
                                        ui.label('⚠️ 危险操作').classes('text-lg font-bold text-red-600')
                                        ui.label(f'确定要永久删除这 {len(target_files)} 张图片吗？').classes('font-bold')
                                        ui.label('包含关联的 .txt 标注文件。此操作无法撤销！').classes('text-sm text-gray-600')
                                        
                                        with ui.row().classes('w-full justify-end mt-4 gap-2'):
                                            ui.button('取消', on_click=d.close).classes(BTN_SECONDARY)
                                            
                                            def confirm_delete():
                                                d.close()
                                                count = 0
                                                for f_path in target_files:
                                                    try:
                                                        if os.path.exists(f_path):
                                                            os.remove(f_path)
                                                            count += 1
                                                        # Try delete txt
                                                        txt_path = os.path.splitext(f_path)[0] + ".txt"
                                                        if os.path.exists(txt_path):
                                                            os.remove(txt_path)
                                                    except Exception as e:
                                                        print(f"Delete error: {e}")
                                                
                                                state.selected_files.clear()
                                                ui.notify(f"已删除 {count} 张图片", type="positive")
                                                refresh_gallery()
                                                
                                            ui.button('确定删除', on_click=confirm_delete).classes(BTN_DANGER)
                                    d.open()
                                    return

                                # 检查是否是清除打标操作，如果是，则弹出确认对话框
                                if op == "清除打标 (Clear Tags)":
                                    with ui.dialog() as d, ui.card():
                                        ui.label('⚠️ 警告').classes('text-lg font-bold text-red-600')
                                        ui.label('确定要清除选中图片的打标信息吗？此操作不可撤销。')
                                        with ui.row().classes('w-full justify-end mt-4'):
                                            ui.button('取消', on_click=d.close).props('flat')
                                            def confirm_clear():
                                                d.close()
                                                start_batch_process()
                                            ui.button('确定清除', on_click=confirm_clear).classes('bg-red-600 text-white')
                                    d.open()
                                else:
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
                                                    # Cleanup thumbnails
                                                    try:
                                                        for root, dirs, files in os.walk(target_path):
                                                            for f in files:
                                                                try:
                                                                    full_path = os.path.join(root, f)
                                                                    mtime = os.path.getmtime(full_path)
                                                                    hash_str = f"{full_path}_{mtime}"
                                                                    thumb_name = hashlib.md5(hash_str.encode('utf-8')).hexdigest() + ".jpg"
                                                                    thumb_path = os.path.join(THUMB_DIR, thumb_name)
                                                                    if os.path.exists(thumb_path):
                                                                        os.remove(thumb_path)
                                                                except: pass
                                                    except: pass

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
                            
                            # Logic: Prioritize Selection > Current Active > Empty
                            target_files = list(state.selected_files)
                            if not target_files and current_active_file:
                                target_files = [current_active_file]
                                
                            count = len(target_files)
                            
                            with details_container:
                                if count == 0:
                                    ui.label("请点击图片或勾选图片以查看详细信息").classes('text-gray-500 italic')
                                    return

                                if count == 1:
                                    f_path = target_files[0]
                                    if not os.path.exists(f_path):
                                        ui.label("文件不存在").classes('text-red-500')
                                        return
                                        
                                    stat = os.stat(f_path)
                                    ctime = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                                    mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                                    size_str = format_size(stat.st_size)
                                    fmt, res = get_image_info(f_path)
                                    dataset_name = os.path.basename(os.path.dirname(f_path))
                                    
                                    with ui.card().classes(CARD_STYLE + ' w-full'):
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
                                    
                                    for f_path in target_files:
                                        if os.path.exists(f_path):
                                            total_size += os.path.getsize(f_path)
                                            _, res = get_image_info(f_path)
                                            resolutions.add(f"{res[0]} x {res[1]}")
                                            dataset_names.add(os.path.basename(os.path.dirname(f_path)))
                                    
                                    ds_display = ", ".join(sorted(list(dataset_names)))
                                    if len(dataset_names) > 3:
                                        ds_display = f"{len(dataset_names)} 个数据集"
                                        
                                    res_display = ", ".join(sorted(list(resolutions)))
                                    
                                    with ui.card().classes(CARD_STYLE + ' w-full'):
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
                        
                        with ui.card().classes(CARD_STYLE + ' mb-6'):
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

                        with ui.card().classes(CARD_STYLE + ' w-full'):
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


    # --- Global Footer (AI Tab Only) ---
                with ui.column().classes('w-full flex-none p-4 bg-white border-t border-gray-200 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] z-50 shrink-0 gap-4').bind_visibility_from(tabs, 'value', value='tab_ai'):
                    with ui.row().classes('w-full items-center relative h-6'):
                        progress_bar = ui.linear_progress(value=0.0, show_value=False).classes('w-full h-6 rounded-full absolute top-0 left-0')
                        progress_label = ui.label('0%').classes('z-10 w-full text-center text-xs font-bold text-white mix-blend-difference')
                    status_label = ui.label('就绪').classes('text-sm text-gray-500')
                    with ui.row().classes('w-full gap-4 items-center'):
                        ui.button('🚀 开始任务', on_click=start_ai_task).classes(BTN_PRIMARY + ' w-full md:w-auto md:flex-1 text-lg h-12 shadow-md')
                        ui.button('⏹ 停止', on_click=stop_worker).classes(BTN_DANGER + ' w-full md:w-32 h-12 shadow-md')

                    # 定时器：轮询后台状态以更新 UI
                    def update_ui_loop():
                        progress_bar.value = state.process_progress
                        progress_label.text = f"{int(state.process_progress * 100)}%"
                        status_label.text = state.process_status
                            
                        if not state.is_processing and state.process_progress >= 1.0 and "完成" in status_label.text:
                            if not getattr(state, 'has_auto_refreshed', False):
                                state.has_auto_refreshed = True
                                refresh_gallery()
                                ui.notify("任务完成，列表已刷新", type="positive")
                                if gallery_container:
                                    gallery_container.update()
                        elif state.is_processing:
                            state.has_auto_refreshed = False
                            
                    ui.timer(0.1, update_ui_loop)

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
    
    # 强制设置初始 tab，触发 visibility bind 更新
    ui.timer(0.1, lambda: tabs.set_value('tab_ai'), once=True)

