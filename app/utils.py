import os
import hashlib
import cv2
from PIL import Image, ImageOps
from app.config import THUMB_DIR

def is_video(path):
    return path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

def get_caption_path(img_path):
    return os.path.splitext(img_path)[0] + ".txt"

def get_caption(img_path):
    txt_path = get_caption_path(img_path)
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""

def save_caption(img_path, content):
    txt_path = get_caption_path(img_path)
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving caption: {e}")

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

def generate_thumbnail(path):
    """
    Generate thumbnail and return its path. 
    Returns None if file not found, or raises exception on failure.
    """
    if not os.path.exists(path):
        return None
    
    # Generate unique thumb name based on path + mtime (to handle updates)
    try:
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
                # Return None on failure to let caller handle it (e.g. fallback to original)
                return None

        return thumb_path
    except Exception as e:
        print(f"Thumbnail outer error: {e}")
        return None
