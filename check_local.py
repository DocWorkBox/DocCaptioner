import torch
import torchvision
import sys
import os

print(f"Python Version: {sys.version}")
print(f"Torch Version: {torch.__version__}")
print(f"TorchVision Version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

try:
    from torchvision import ops
    print("Testing NMS operator...")
    # Simple NMS test
    if torch.cuda.is_available():
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]]).cuda()
        scores = torch.tensor([1.0]).cuda()
    else:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([1.0])
    ops.nms(boxes, scores, 0.5)
    print("NMS Operator: OK")
except Exception as e:
    print(f"NMS Operator Failed: {e}")
    sys.exit(1)

print("\n--- Optional Dependencies ---")
try:
    import bitsandbytes
    print(f"bitsandbytes: Installed (v{bitsandbytes.__version__}) - 8-bit/4-bit Quantization Supported")
except ImportError:
    print("bitsandbytes: Not Installed (Quantization unavailable)")
except Exception as e:
    print(f"bitsandbytes: Import Failed ({e}) - Quantization unavailable")

import importlib.util
if importlib.util.find_spec("flash_attn"):
    print("flash_attn: Installed - Flash Attention 2 Supported")
else:
    print("flash_attn: Not Installed - Flash Attention 2 unavailable")

print("\nStandard Environment Check Passed.")
