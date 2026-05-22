"""Small demo for the minimal spectral-correlation resampling detector.

Usage:
    python3 demo_resampling_detection.py
    python3 demo_resampling_detection.py path/to/image.png

With no image path, the script creates a synthetic resampling example from
``skimage.data.camera``. It prints the distances with the smallest NFAs.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from skimage import data
from skimage.transform import resize

from resampling_core import DetectionResult, detect_both_axes


def load_image(path: str | None) -> np.ndarray:
    if path is None:
        original = data.camera()
        return resize(
            original,
            (384, 384),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.uint8)

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def print_top_distances(result: DetectionResult, top_k: int = 8) -> None:
    order = np.argsort(result.nfa)[:top_k]
    axis_name = "vertical/rows" if result.axis == 0 else "horizontal/cols"
    print(f"\nAxis: {axis_name}")
    print("distance  maxima_count  NFA          log10(NFA)")
    for index in order:
        print(
            f"{int(result.distances[index]):8d}  "
            f"{int(result.maxima_counts[index]):12d}  "
            f"{result.nfa[index]:.3e}  "
            f"{result.log10_nfa[index]:10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="optional image path")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    args = parser.parse_args()

    image = load_image(args.image)
    vertical, horizontal = detect_both_axes(
        image,
        tv_weight=args.tv_weight,
        radius=args.radius,
    )

    print(f"Image shape: {image.shape}")
    print(f"Patch shape: {vertical.patch_shape}")
    print_top_distances(vertical)
    print_top_distances(horizontal)


if __name__ == "__main__":
    main()
