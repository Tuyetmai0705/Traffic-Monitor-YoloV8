"""Inference device discovery (CUDA GPUs or CPU)."""

from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger("device")


def _cuda_names() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except (ImportError, RuntimeError):
        pass
    return []


def available_devices() -> list[tuple[str, str]]:
    """Return ``(device_id, label)`` pairs, GPUs first, e.g. ``("cuda:0", "GPU 0 · NVIDIA …")``."""
    devices = [(f"cuda:{index}", f"GPU {index} · {name}") for index, name in enumerate(_cuda_names())]
    devices.append(("cpu", "CPU"))
    return devices


def resolve_device(requested: str) -> str:
    """Return ``requested`` when usable, otherwise fall back to ``cpu``."""
    if not requested.lower().startswith("cuda"):
        return "cpu"
    names = _cuda_names()
    try:
        index = int(requested.split(":")[1]) if ":" in requested else 0
    except ValueError:
        index = -1
    if 0 <= index < len(names):
        return f"cuda:{index}"
    logger.warning("CUDA device '%s' not available, falling back to CPU", requested)
    return "cpu"


def describe_device(device: str) -> str:
    """Human readable device name, e.g. ``cuda:0 (NVIDIA GeForce MX570 A)``."""
    if device.startswith("cuda"):
        names = _cuda_names()
        try:
            return f"{device} ({names[int(device.split(':')[1])]})"
        except (IndexError, ValueError):
            return device
    return "CPU"
