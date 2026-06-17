#!/usr/bin/env python3
"""Replace broken RAISE TIFF entries in sample_list.csv with valid RAISE-2k images.

Legacy utility for the old RAISE-1000-ms ``sample_list.csv`` workflow (v1 mask
data curation). **Not used** in the n6 main experiment path (see
``docs/EXPERIMENT_RUNBOOK.md`` — split comes from ``split_raise.py``).

Quality checks use a fixed 512×512 RGB crop and the **native** log-rFFT spectrum
at that resolution (512×257), matching the current ``spectrum.py`` API.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.make_patches import random_crop_rgb
from src.processing.residuals import tv_residual
from src.processing.spectrum import compute_log_rfft_spectrum
from src.processing.transforms import apply_jpeg_pil, resize_pil, rgb_to_y_float

RAISE_2K_CSV_URL = "http://loki.disi.unitn.it/RAISE/getFile.php?p=2k"
DEFAULT_PATCH_SIZE = 512
DEFAULT_PATCHES = 10
DEFAULT_JPEG_QUALITY = 80
DEFAULT_DOWNSAMPLE_FACTORS = (2, 4, 8, 16)


def fetch_raise_2k_csv(cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    with urllib.request.urlopen(RAISE_2K_CSV_URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("RAISE-2k zip does not contain a CSV file")
        zf.extract(names[0], cache_path.parent)
        extracted = cache_path.parent / names[0]
        extracted.rename(cache_path)
    return cache_path


def load_raise_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filename_from_row(row: dict[str, str]) -> str:
    name = row["File"]
    return name if name.upper().endswith(".TIF") else f"{name}.TIF"


def try_load_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.width < DEFAULT_PATCH_SIZE or image.height < DEFAULT_PATCH_SIZE:
                return None
            return image.convert("RGB")
    except Exception:
        return None


def download_image(url: str, dst: Path) -> Image.Image | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size < 1000:
        urllib.request.urlretrieve(url, dst)
    rgb = try_load_image(dst)
    if rgb is None and dst.exists():
        dst.unlink(missing_ok=True)
    return rgb


def count_zero_spectra(image: Image.Image, seed: int, patches: int) -> int:
    rng = random.Random(seed)
    zero_count = 0
    class_ops: list[tuple[str, object]] = [("original", None)]
    class_ops.append(("JPEG", DEFAULT_JPEG_QUALITY))
    for factor in DEFAULT_DOWNSAMPLE_FACTORS:
        class_ops.append((f"downsample_x{factor}", factor))

    for _ in range(patches):
        patch = random_crop_rgb(image, DEFAULT_PATCH_SIZE, rng)
        for _, op in class_ops:
            if op is None:
                processed = patch
            elif isinstance(op, int) and op == DEFAULT_JPEG_QUALITY:
                processed = apply_jpeg_pil(patch, op)
            else:
                side = DEFAULT_PATCH_SIZE // int(op)
                processed = resize_pil(patch, side, "bicubic")
            y = rgb_to_y_float(processed)
            residual = tv_residual(y, weight=0.08, max_num_iter=30)
            spectrum = compute_log_rfft_spectrum(residual, dc_sigma_bins=3.0)
            if not spectrum.any():
                zero_count += 1
    return zero_count


def quality_from_zero_count(zero_count: int, load_status: str) -> tuple[str, str]:
    if load_status != "ok":
        return "BAD", "decoder error"
    if zero_count > 0:
        return (
            "DEGRADED",
            f"produced {zero_count} all-zero spectra (flat patch / processing degeneracy)",
        )
    return "OK", ""


def pick_replacement(
    candidates: list[tuple[str, str, str]],
    download_dir: Path,
    seed: int,
    max_attempts: int,
) -> tuple[str, str, str, int]:
    rng = random.Random(seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    for filename, url, split in shuffled[:max_attempts]:
        dst = download_dir / filename
        image = download_image(url, dst)
        if image is None:
            continue
        zero_count = count_zero_spectra(image, seed=seed, patches=DEFAULT_PATCHES)
        return filename, url, split, zero_count
    raise RuntimeError(f"No valid replacement found within {max_attempts} candidates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-list",
        default="data/raw/RAISE-1000-ms/sample_list.csv",
        type=Path,
    )
    parser.add_argument(
        "--raise-2k-csv",
        default="data/raw/raise_2k.csv",
        type=Path,
        help="Cached RAISE-2k metadata CSV (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--download-dir",
        default="data/raw/raise_tiff",
        type=Path,
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args()

    sample_path = args.sample_list
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path
    download_dir = args.download_dir if args.download_dir.is_absolute() else ROOT / args.download_dir
    raise_2k_csv = args.raise_2k_csv if args.raise_2k_csv.is_absolute() else ROOT / args.raise_2k_csv

    df = pd.read_csv(sample_path)
    bad_rows = df[df["load_status"] == "load_error"]
    if bad_rows.empty:
        print("No load_error rows found; nothing to resample.")
        return

    fetch_raise_2k_csv(raise_2k_csv)
    used = set(df["filename"])
    raise_rows = load_raise_rows(raise_2k_csv)
    candidates: list[tuple[str, str, str]] = []
    for row in raise_rows:
        filename = filename_from_row(row)
        if filename in used:
            continue
        candidates.append((filename, row["TIFF"], str(bad_rows.iloc[0]["splits"])))

    replacement_name, replacement_url, split, zero_count = pick_replacement(
        candidates,
        download_dir=download_dir,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    load_status = "ok"
    quality, reason = quality_from_zero_count(zero_count, load_status)

    removed = bad_rows["filename"].tolist()
    df = df[df["load_status"] != "load_error"].copy()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "filename": replacement_name,
                        "quality": quality,
                        "reason": reason,
                        "zero_spectrum_count": zero_count,
                        "load_status": load_status,
                        "splits": split,
                        "path": f"data/raw/raise_tiff/{replacement_name}",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df = df.sort_values(
        by=["zero_spectrum_count", "filename"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    df.to_csv(sample_path, index=False)

    print(f"Removed: {', '.join(removed)}")
    print(f"Added: {replacement_name} ({quality}, zero_spectrum_count={zero_count}, split={split})")
    print(f"Updated: {sample_path}")


if __name__ == "__main__":
    main()
