"""Compute-device selection for the torch stages (Cellpose): CUDA (workstation RTX 5090, Alpine
A100 / L40), Apple Silicon via Metal (MPS), or CPU.

Order: ``GERMQUANT_DEVICE`` (cuda | mps | cpu | auto) or the config key
``segmentation.nuclei.device`` when set, else the first available of CUDA, MPS, CPU. On MPS the
PyTorch fallback for unsupported operators is switched on (``PYTORCH_ENABLE_MPS_FALLBACK=1``) so an
operator Metal lacks runs on the CPU instead of raising, and models run in float32 (bfloat16 support
on MPS is incomplete). SpotMAX runs on the CPU on every platform (its optional CuPy path is CUDA-only
and is not used).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_CHOICES = ("auto", "cuda", "mps", "cpu")


def available() -> dict[str, bool]:
    """Which torch devices this machine offers (False everywhere when torch is missing)."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - CPU-only environment without torch
        return {"cuda": False, "mps": False}
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {"cuda": cuda, "mps": mps}


def select_device(prefer: str | None = None) -> str:
    """'cuda', 'mps' or 'cpu'. `prefer` (config) is overridden by $GERMQUANT_DEVICE; 'auto' or None
    picks the first available accelerator. Asking for an accelerator that is not available falls back
    to the next one with a warning, never an error."""
    want = (os.environ.get("GERMQUANT_DEVICE") or prefer or "auto").strip().lower()
    if want not in _CHOICES:
        log.warning("unknown device %r (choices %s); using auto", want, _CHOICES)
        want = "auto"
    have = available()
    order = {"auto": ["cuda", "mps"], "cuda": ["cuda", "mps"], "mps": ["mps", "cuda"], "cpu": []}[want]
    for dev in order:
        if have.get(dev):
            if want not in ("auto", dev):
                log.warning("device %r not available; using %s", want, dev)
            if dev == "mps":
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return dev
    if want not in ("auto", "cpu"):
        log.warning("device %r not available; using cpu", want)
    return "cpu"


def torch_device(name: str):
    import torch

    return torch.device(name)


def describe() -> str:
    """One line for logs and `check-gpu`."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f"torch not importable ({e})"
    have = available()
    if have["cuda"]:
        cap = torch.cuda.get_device_capability()
        return f"torch {torch.__version__}  cuda {torch.cuda.get_device_name(0)} (compute capability {cap[0]}.{cap[1]})"
    if have["mps"]:
        return f"torch {torch.__version__}  mps (Apple Silicon GPU via Metal)"
    return f"torch {torch.__version__}  cpu only"
