from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def setup_device_env(config: Dict[str, Any]) -> None:
    training = config.get("training", {})
    device_name = str(training.get("device", "cpu")).lower()
    if device_name == "npu":
        visible = training.get("npu_visible_devices", training.get("ascend_rt_visible_devices", "0"))
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(visible))
    elif device_name == "cuda":
        visible = training.get("cuda_visible_devices")
        if visible is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(visible))


def get_device(config: Dict[str, Any]) -> torch.device:
    requested = str(config["training"]["device"]).lower()
    if requested == "npu":
        if _npu_available():
            return torch.device("npu:0")
        print("NPU requested but torch_npu is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def supports_amp(device: torch.device, config: Dict[str, Any]) -> bool:
    if not bool(config["training"].get("amp", False)):
        return False
    return device.type in ("cuda", "npu")


def use_pin_memory(device: torch.device, config: Dict[str, Any]) -> bool:
    if device.type not in ("cuda", "npu"):
        return False
    return bool(config["training"].get("pin_memory", True))


@contextmanager
def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        yield
        return
    if device.type == "cuda":
        from torch.cuda.amp import autocast

        with autocast():
            yield
    elif device.type == "npu":
        with torch.npu.amp.autocast():
            yield
    else:
        yield


def create_grad_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type not in ("cuda", "npu"):
        return None
    if device.type == "cuda":
        from torch.cuda.amp import GradScaler

        return GradScaler()
    from torch.npu.amp import GradScaler

    return GradScaler()


def print_device_info() -> None:
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print("npu available:", _npu_available())
    if _npu_available():
        print("npu device count:", torch.npu.device_count())
