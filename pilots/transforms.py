from __future__ import annotations

import io

import numpy as np
from PIL import Image
from skimage.transform import resize

from .config import INTERP_ORDER, TARGET_SIZE


def resize_gray(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image.astype(np.uint8, copy=False)
    out = resize(
        image,
        (size, size),
        order=INTERP_ORDER,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"))


def resample_via_reference(
    square_source: np.ndarray,
    reference_size: int,
    target_size: int = TARGET_SIZE,
) -> tuple[np.ndarray, int]:
    """Down/up through ``reference_size`` then to ``target_size`` (bicubic)."""
    reference = resize_gray(square_source, reference_size)
    target = resize_gray(reference, target_size)
    period = abs(square_source.shape[0] - target_size) % target_size
    if period == 0 and square_source.shape[0] != target_size:
        period = abs(square_source.shape[0] - target_size)
    return target, period


def simulate_x8_block_resample(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Resize to 1/8 and back to mimic strong 8-pixel grid artefacts."""
    small = max(target_size // 8, 8)
    down = resize_gray(image, small)
    return resize_gray(down, target_size)


def reference_sizes_for_group(
    reference_peak: int,
    k: int,
    target_size: int,
    pattern: str,
) -> list[int]:
    if pattern == "ref_plus_k":
        base = reference_peak
    elif pattern == "target_minus_ref_plus_k":
        base = target_size - reference_peak
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    sizes = [base + k * offset for offset in (0, 1, 2)]
    return [max(16, min(target_size - 1, size)) for size in sizes]
