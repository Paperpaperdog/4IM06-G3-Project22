#!/usr/bin/env python3
"""
Preprocess RAISE images:
1) Convert TIFF/TIF to PNG
2) Center-crop to square using the smaller side

Example:
    python scripts/preprocess_raise.py --input-dir data/RAISE1K --output-dir data/RAISE1K_png_square
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def center_crop_to_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return img.crop((left, top, right, bottom))


def process_one(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as img:
        # Normalize mode for PNG output.
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        square = center_crop_to_square(img)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        square.save(dst_path, format="PNG")


def collect_inputs(input_dir: Path) -> list[Path]:
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files)


def target_path(src: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = src.relative_to(input_dir)
    return (output_dir / rel).with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert images to PNG and center-crop to square."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input root folder")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = collect_inputs(input_dir)
    if not files:
        print("No supported image files found.")
        return

    ok = 0
    failed = 0
    for src in files:
        dst = target_path(src, input_dir, output_dir)
        try:
            process_one(src, dst)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {src}: {exc}")

    print(f"Done. Success: {ok}, Failed: {failed}, Total: {len(files)}")


if __name__ == "__main__":
    main()
