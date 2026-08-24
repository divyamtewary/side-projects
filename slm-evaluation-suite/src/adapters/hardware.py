from __future__ import annotations
import platform
import sys
from datetime import datetime, timezone
import psutil

def capture_environment() -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    vm = psutil.virtual_memory()
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except Exception:
        torch_version = None
        cuda_available = False
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import psutil as _ps
        psutil_version = _ps.__version__
    except Exception:
        psutil_version = None

    return {
        "schema_version": "1.0",
        "timestamp_utc": now,
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "psutil_version": psutil_version,
        "cuda_available": bool(cuda_available),
        "truth_level": "measured",
    }
