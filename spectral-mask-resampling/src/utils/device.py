"""Device resolution shared by train/evaluate/visualize.

Supports CUDA GPUs and Ascend NPUs (via torch_npu) so the mask pipeline can run
on the same supercomputer compute nodes as the CNN pipeline, with a CPU
fallback for local smoke tests.
"""

from __future__ import annotations

import os


def _npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def setup_device_env(config: dict) -> None:
    training = config.get("training", {})
    device_name = str(training.get("device", "cpu")).lower()
    if device_name == "npu":
        visible = training.get("npu_visible_devices", training.get("ascend_rt_visible_devices", "0"))
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(visible))
    elif device_name == "cuda":
        visible = training.get("cuda_visible_devices")
        if visible is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(visible))


def resolve_device(requested: str):
    import torch

    requested = str(requested).lower()
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
    return torch.device(requested)
