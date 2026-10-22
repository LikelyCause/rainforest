"""Detect an NVIDIA GPU and whether this image's XGBoost can train on it.

Detection runs once per process and never raises: no driver, no `nvidia-smi`, a CPU-only XGBoost
build or a failing probe all just mean "no usable GPU". XGBoost silently falls back to the CPU
when it can't reach a GPU, so the probe fits a tiny model and reads back the device it really used.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_lock = threading.Lock()
_cached: dict[str, Any] | None = None
_probed = False


def _nvidia_smi() -> dict[str, Any] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("nvidia-smi present but failed: %s", exc)
        return None
    line = next((ln for ln in out.splitlines() if ln.strip()), None)
    if line is None:
        return None
    name, mem, driver = (p.strip() for p in (line.split(",") + ["", ""])[:3])
    try:
        memory_mb = int(float(mem))
    except ValueError:
        memory_mb = None
    return {"name": name, "memory_mb": memory_mb, "driver": driver or None, "count": len(out.strip().splitlines())}


def xgboost_cuda_build() -> bool:
    try:
        import xgboost

        return bool(xgboost.build_info().get("USE_CUDA"))
    except Exception:  # noqa: BLE001 - not installed or a broken install
        return False


def effective_device(booster: Any) -> str:
    """The device a fitted XGBoost booster actually used ("cpu" after a silent fallback)."""
    try:
        return json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    except Exception:  # noqa: BLE001
        return "unknown"


_PROBE = """
import numpy as np
from xgboost import XGBClassifier
from app.gpu import effective_device
X = np.arange(64, dtype=np.float32).reshape(32, 2)
m = XGBClassifier(n_estimators=1, max_depth=1, device="cuda", tree_method="hist", verbosity=0)
print(effective_device(m.fit(X, (X[:, 0] > 31).astype(int)).get_booster()))
"""


def _probe_xgboost() -> str | None:
    """None if a CUDA fit works, else why not.

    Runs in a subprocess so a driver fault can't take down the API worker and the probe's CUDA
    context doesn't hold GPU memory for the life of the process.
    """
    if not xgboost_cuda_build():
        return "this image's XGBoost is the CPU build (rebuild with GPU=1, see docs/deployment.md › GPU)"
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"XGBoost CUDA test fit failed: {exc}"
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["no output"])[-1]
        return f"XGBoost CUDA test fit failed: {tail}"
    device = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "unknown"
    return None if device.startswith("cuda") else f"XGBoost fell back to {device} (CUDA 13 builds need driver 580+)"


def detect(refresh: bool = False) -> dict[str, Any] | None:
    """{available, name, memory_mb, driver, count, reason} for the first GPU, or None when there is none."""
    global _cached, _probed
    with _lock:
        if _probed and not refresh:
            return _cached
        info = None
        try:
            info = _nvidia_smi()
            if info is not None:
                reason = _probe_xgboost()
                info = {**info, "available": reason is None, "reason": reason}
        except Exception:  # noqa: BLE001
            log.exception("GPU detection failed")
            info = None
        _cached, _probed = info, True
        return info


def warm_up() -> threading.Thread:
    """Run detection in the background at startup; the probe's test fit can take seconds, and the first
    /api/system call would otherwise wait for it (it still waits, on the lock, if detection hasn't finished)."""
    t = threading.Thread(target=detect, name="gpu-detect", daemon=True)
    t.start()
    return t


def gpu_available() -> bool:
    info = detect()
    return bool(info and info["available"])
