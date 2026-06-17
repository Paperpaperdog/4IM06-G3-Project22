"""Unified observation protocol shared by all three forensic routes.

This module is the SINGLE SOURCE OF TRUTH for:

1. The canonical 6-class processing-history task (now including upsampling).
2. The shared set of observed image sizes used for the input-size sweep.
3. How an observed ``o x o`` RGB patch is produced for every class, so that the
   classical detector, the spectral mask classifier and the spectral CNN all
   consume *identical* image inputs at every size.

Design choices (n6 protocol):

* Classes are symmetric around "no rescale":

  ===================  ================================================
  class                how the observed o x o patch is produced
  ===================  ================================================
  original             crop o x o directly
  JPEG_Q80             crop o x o, JPEG encode/decode at quality 80
  downsample_x8        crop (8o) x (8o),    bicubic resize DOWN to o
  downsample_x16       crop (16o) x (16o),  bicubic resize DOWN to o
  upsample_x4          crop (o/4) x (o/4),  bicubic resize UP to o
  upsample_x8          crop (o/8) x (o/8),  bicubic resize UP to o
  ===================  ================================================

* Observed sizes: ``{32, 64, 96, 128}``.  Each is trained as a SEPARATE model
  per route, with its own native rFFT spectrum (no shared frequency grid).
  ``16 * 128 = 2048`` still fits a RAISE TIFF, so every (class, size) pair is
  realizable; ``upsample_x8`` at o=32 crops 4 px (the minimum allowed).

The two deep-learning subprojects keep their own spectrum pipelines; they only
import :func:`make_observed_patch` / :func:`required_source_crop` so the *image*
generation is guaranteed to match across routes.
"""

from __future__ import annotations

import io
from typing import Callable, Optional, TypeVar

import numpy as np
from PIL import Image

CANONICAL_CLASSES = [
    "original",
    "JPEG_Q80",
    "downsample_x8",
    "downsample_x16",
    "upsample_x4",
    "upsample_x8",
]

OBSERVED_SIZES = [32, 64, 96, 128]

DEFAULT_SEED = 123
DEFAULT_JPEG_QUALITY = 80
MIN_UPSAMPLE_CROP = 4

SPLIT_NAMES = ("train", "val", "test")

_UPSAMPLE_FACTORS = {"upsample_x4": 4, "upsample_x8": 8}
_DOWNSAMPLE_FACTORS = {"downsample_x8": 8, "downsample_x16": 16}

_BICUBIC = Image.Resampling.BICUBIC

SourceEntry = TypeVar("SourceEntry")


def normalize_split_data(split_data: dict) -> dict[str, list]:
    """Accept mask-style ``{train: [...]}`` or CNN-style ``{splits: {train: [...]}}``."""
    if "splits" in split_data:
        return {name: list(split_data["splits"][name]) for name in SPLIT_NAMES}
    return {name: list(split_data[name]) for name in SPLIT_NAMES}


def per_sample_seed(
    global_seed: int,
    split: str,
    observed_size: int,
    class_index: int,
    sample_index: int,
) -> int:
    """Deterministic seed for one (split, size, class, sample_index) draw.

    Mask and CNN must call this with the same arguments so random source choice
    and crop coordinates match pixel-for-pixel.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    split_id = SPLIT_NAMES.index(split)
    return (
        int(global_seed)
        + split_id * 10_000_003
        + int(observed_size) * 100_007
        + int(class_index) * 1_009
        + int(sample_index) * 7_919
    )


def sample_numpy_rng(
    global_seed: int,
    split: str,
    observed_size: int,
    class_index: int,
    sample_index: int,
) -> np.random.Generator:
    return np.random.default_rng(
        per_sample_seed(global_seed, split, observed_size, class_index, sample_index)
    )


def class_index(class_name: str) -> int:
    return CANONICAL_CLASSES.index(class_name)


def required_source_crop(class_name: str, observed_size: int) -> int:
    """Source crop side length needed to realize ``class_name`` at ``observed_size``.

    A sample is only feasible if the source image is at least this large.
    """
    if class_name in ("original", "JPEG_Q80"):
        return int(observed_size)
    if class_name in _UPSAMPLE_FACTORS:
        factor = _UPSAMPLE_FACTORS[class_name]
        if observed_size % factor != 0:
            raise ValueError(
                f"observed_size {observed_size} not divisible by upsample factor {factor}"
            )
        crop = int(observed_size // factor)
        if crop < MIN_UPSAMPLE_CROP:
            raise ValueError(
                f"Upsample class '{class_name}' with observed_size={observed_size} yields a "
                f"degenerate crop of {crop}px (< {MIN_UPSAMPLE_CROP})."
            )
        return crop
    if class_name in _DOWNSAMPLE_FACTORS:
        return int(observed_size * _DOWNSAMPLE_FACTORS[class_name])
    raise ValueError(f"Unknown class '{class_name}'. Expected one of {CANONICAL_CLASSES}.")


def _apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def transform_source_patch(
    patch: Image.Image,
    class_name: str,
    observed_size: int,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Image.Image:
    """Turn an already-cropped source patch into the final ``o x o`` RGB image.

    ``patch`` must be ``required_source_crop(class_name, observed_size)`` on a side.
    """
    patch = patch.convert("RGB")
    observed_size = int(observed_size)

    if class_name == "original":
        out = patch
    elif class_name == "JPEG_Q80":
        out = _apply_jpeg(patch, jpeg_quality)
    elif class_name in _UPSAMPLE_FACTORS or class_name in _DOWNSAMPLE_FACTORS:
        out = patch.resize((observed_size, observed_size), resample=_BICUBIC)
    else:
        raise ValueError(f"Unknown class '{class_name}'.")

    if out.size != (observed_size, observed_size):
        out = out.resize((observed_size, observed_size), resample=_BICUBIC)
    return out


def make_observed_patch(
    source_img: Image.Image,
    class_name: str,
    observed_size: int,
    rng,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Optional[Image.Image]:
    """Produce one observed ``o x o`` RGB patch for ``class_name``.

    ``rng`` must expose ``integers(low, high)`` (``numpy.random.Generator``) or
    ``randint(low, high_inclusive)`` (``random.Random``).  Returns ``None`` if the
    source image is too small for the required crop.
    """
    crop_size = required_source_crop(class_name, observed_size)
    width, height = source_img.size
    if width < crop_size or height < crop_size:
        return None

    max_x = width - crop_size
    max_y = height - crop_size
    if hasattr(rng, "integers"):
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    else:
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0

    patch = source_img.crop((x, y, x + crop_size, y + crop_size))
    return transform_source_patch(patch, class_name, observed_size, jpeg_quality)


def make_aligned_observed_patch(
    source_entries: list[SourceEntry],
    open_image: Callable[[SourceEntry], Image.Image | None],
    class_name: str,
    observed_size: int,
    global_seed: int,
    split: str,
    class_index: int,
    sample_index: int,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Optional[Image.Image]:
    """Generate one aligned RGB patch shared by Mask and CNN preprocessors.

    Uses a fixed per-sample seed, then tries sources in a deterministic rotation
    if the first pick is too small or unreadable.
    """
    n = len(source_entries)
    if n == 0:
        return None

    rng = sample_numpy_rng(global_seed, split, observed_size, class_index, sample_index)
    start_idx = int(rng.integers(0, n))
    for k in range(n):
        entry = source_entries[(start_idx + k) % n]
        try:
            img = open_image(entry)
            if img is None:
                continue
            patch = make_observed_patch(img, class_name, observed_size, rng, jpeg_quality)
            if patch is not None:
                return patch
        except Exception:
            continue
    return None


def summary() -> str:
    lines = ["Unified protocol", f"classes={CANONICAL_CLASSES}", f"observed_sizes={OBSERVED_SIZES}", ""]
    for o in OBSERVED_SIZES:
        crops = {c: required_source_crop(c, o) for c in CANONICAL_CLASSES}
        lines.append(f"o={o:>3}: " + ", ".join(f"{c}->{s}" for c, s in crops.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
