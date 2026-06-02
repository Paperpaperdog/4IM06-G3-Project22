from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .config import RAISE_PNG_DIR, RAISE_RAW_DIR, REPO_ROOT
from .io_utils import (
    crop_center_square,
    fallback_source_images,
    load_grayscale,
    save_grayscale_png,
    write_manifest,
)
from .raise_index import (
    default_csv_path,
    download_tiffs_from_csv,
    find_local_tiffs,
    load_raise_index,
)
from .transforms import resize_gray


def convert_tiff_paths_to_png(
    tiff_files: list[Path],
    out_dir: Path,
    min_side: int,
    index_by_id: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for index, tiff_path in enumerate(tiff_files):
        image = np.asarray(Image.open(tiff_path))
        square = crop_center_square(image)
        if square.shape[0] < min_side:
            continue
        stem = tiff_path.stem
        out_path = out_dir / f"{stem}.png"
        save_grayscale_png(out_path, square)

        meta = (index_by_id or {}).get(stem, {})
        rows.append(
            {
                "image_id": stem,
                "source": "raise_tiff",
                "source_path": str(tiff_path),
                "png_path": str(out_path),
                "original_side": str(square.shape[0]),
                "device": meta.get("Device", ""),
                "keywords": meta.get("Keywords", ""),
                "image_size": meta.get("Image Size", ""),
            }
        )
        print(f"[{index + 1}/{len(tiff_files)}] {tiff_path.name} -> {out_path.name}")
    return rows


def index_by_file_id(csv_path: Path) -> dict[str, dict[str, str]]:
    return {row["File"].strip(): row for row in load_raise_index(csv_path)}


def build_fallback_subset(out_dir: Path, max_images: int) -> list[dict]:
    rows: list[dict] = []
    for name, gray in fallback_source_images(REPO_ROOT, max_images):
        square = crop_center_square(gray)
        out_path = out_dir / f"{name}.png"
        save_grayscale_png(out_path, square)
        rows.append(
            {
                "image_id": name,
                "source": "fallback",
                "source_path": f"builtin_or_img/{name}",
                "png_path": str(out_path),
                "original_side": str(square.shape[0]),
            }
        )
        print(f"fallback: {name} -> {out_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare square PNG subset (RAISE TIFF or fallback).")
    parser.add_argument("--raise-dir", type=Path, default=RAISE_RAW_DIR, help="Folder with RAISE TIFF / RAISE_1k.csv.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="RAISE index CSV (default: <raise-dir>/RAISE_1k.csv if present).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download TIFFs from CSV URLs into <raise-dir>/tiff/ before converting.",
    )
    parser.add_argument("--out-dir", type=Path, default=RAISE_PNG_DIR)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--target-preview", type=int, default=0, help="If >0, also save 384 previews.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    csv_path = args.csv or default_csv_path(args.raise_dir)
    tiff_dir = args.raise_dir / "tiff"

    if args.download:
        if csv_path is None:
            raise SystemExit(f"--download requires RAISE_1k.csv in {args.raise_dir}")
        print(f"Downloading up to {args.max_images} TIFFs from {csv_path.name} ...")
        download_tiffs_from_csv(csv_path, tiff_dir, args.max_images)

    local_tiffs = find_local_tiffs(args.raise_dir)
    if local_tiffs:
        local_tiffs = local_tiffs[: args.max_images]
        meta = index_by_file_id(csv_path) if csv_path else None
        rows = convert_tiff_paths_to_png(local_tiffs, args.out_dir, args.min_side, meta)
    else:
        if csv_path is not None:
            print(
                f"Found {csv_path.name} ({len(load_raise_index(csv_path))} entries) but no local TIFF.\n"
                f"Run with --download, or place .TIF files under {args.raise_dir}/tiff/\n"
                f"Falling back to demo images."
            )
        else:
            print(f"No TIFF/CSV in {args.raise_dir}; using fallback images.")
        rows = build_fallback_subset(args.out_dir, args.max_images)

    if args.target_preview > 0:
        preview_dir = args.out_dir / "preview_384"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            gray = load_grayscale(Path(row["png_path"]))
            preview = resize_gray(gray, args.target_preview)
            preview_path = preview_dir / f"{row['image_id']}_384.png"
            save_grayscale_png(preview_path, preview)
            row["preview_384"] = str(preview_path)

    write_manifest(rows)
    print(f"Wrote {len(rows)} entries to manifest.")


if __name__ == "__main__":
    main()
