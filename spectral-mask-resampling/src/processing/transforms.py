from io import BytesIO

import numpy as np
from PIL import Image


def apply_jpeg_pil(img: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_down_up_pil(img: Image.Image, factor: int, interpolation: str) -> Image.Image:
    if interpolation != "bicubic":
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    width, height = img.size
    resample = Image.Resampling.BICUBIC
    small = img.resize((width // factor, height // factor), resample=resample)
    return small.resize((width, height), resample=resample)


def rgb_to_y_float(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y.astype(np.float32)
