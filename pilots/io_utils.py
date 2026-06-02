from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data as skdata

from .config import MANIFEST_CSV, RAISE_PNG_DIR


def crop_center_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top : top + side, left : left + side]


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def save_grayscale_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.round(image), 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest() -> list[dict]:
    if not MANIFEST_CSV.is_file():
        return []
    with MANIFEST_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_png_subset() -> list[Path]:
    if not RAISE_PNG_DIR.is_dir():
        return []
    return sorted(RAISE_PNG_DIR.glob("*.png"))


def fallback_source_images(repo_root: Path, max_images: int) -> list[tuple[str, np.ndarray]]:
    """Build a small subset when RAISE TIFFs are not available."""
    sources: list[tuple[str, np.ndarray]] = []

    img_dir = repo_root / "img"
    if img_dir.is_dir():
        for path in sorted(img_dir.glob("*.png"))[:max_images]:
            sources.append((path.stem, load_grayscale(path)))

    builtin = {
        "camera": skdata.camera(),
        "astronaut": skdata.astronaut(),
        "chelsea": skdata.chelsea(),
    }
    for name, array in builtin.items():
        if len(sources) >= max_images:
            break
        if array.ndim == 3:
            from skimage.color import rgb2gray

            gray = (rgb2gray(array[..., :3]) * 255).astype(np.uint8)
        else:
            gray = array.astype(np.uint8)
        sources.append((name, gray))

    return sources[:max_images]
