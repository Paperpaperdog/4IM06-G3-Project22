from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image


PIL_INTERPOLATION = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}


def apply_jpeg_pil(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return decoded.copy()


def resize_to_final(img: Image.Image, final_size: int, interpolation: str = "bicubic") -> Image.Image:
    if interpolation not in PIL_INTERPOLATION:
        raise ValueError(f"Unknown interpolation '{interpolation}'. Choose from {sorted(PIL_INTERPOLATION)}.")
    return img.convert("RGB").resize((int(final_size), int(final_size)), PIL_INTERPOLATION[interpolation])


def random_crop(img: Image.Image, crop_size: int, rng: np.random.Generator) -> Optional[tuple[Image.Image, int, int]]:
    width, height = img.size
    crop_size = int(crop_size)
    if width < crop_size or height < crop_size:
        return None
    max_x = width - crop_size
    max_y = height - crop_size
    x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    return img.crop((x, y, x + crop_size, y + crop_size)).convert("RGB"), x, y


def rgb_to_y_float(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y.astype(np.float32)
