import sys
import warnings

# 忽略 bitsandbytes 的 MatMul8bitLt 警告 (已知无害且无法消除)
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
warnings.filterwarnings("ignore", module="bitsandbytes.autograd._functions")

# Print startup banner (English to avoid encoding issues)
print("-" * 50)
print("Initializing DocCaptioner System (v1.1 Modular)...")
print("Loading dependencies (NiceGUI, PyTorch, Transformers)...")
print("First launch may take some time, please wait...")
print("-" * 50)

from nicegui import ui, app
from fastapi.responses import FileResponse
from app.config import DATASET_ROOT
from app.utils import generate_thumbnail
from ui.layout import create_ui

# --- NiceGUI App Setup ---
app.add_static_files('/datasets', DATASET_ROOT)

@app.get('/api/thumbnail')
def get_thumbnail_api(path: str):
    thumb_path = generate_thumbnail(path)
    if thumb_path:
        return FileResponse(thumb_path)
    return 500

@ui.page('/')
def index():
    create_ui()

ui.run(title="DocCaptioner v1.1", host="127.0.0.1", port=9090, reload=True, dark=False, storage_secret="secret")
