"""Verify shared n6 spectrum cache exists and has expected layout.

Mask and CNN read the **same** directory (`data/processed/n6_spectra_size{N}/`).
This script checks files/shapes; it does not compare two separate caches.

Usage:
  cd 4IM06-G3-Project22
  python scripts/verify_spectral_alignment.py --size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / "data" / "processed" / f"n6_spectra_size{args.size}"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache missing: {cache_dir}\n"
            f"Run: SIZE={args.size} bash scripts/prepare_n6_spectra.sh"
        )

    spec_path = cache_dir / f"{args.split}_spectra.npy"
    lab_path = cache_dir / f"{args.split}_labels.npy"
    if not spec_path.exists() or not lab_path.exists():
        raise FileNotFoundError(f"Missing {args.split} arrays under {cache_dir}")

    spectra = np.load(spec_path, mmap_mode="r")
    labels = np.load(lab_path, mmap_mode="r")
    o = args.size
    expected_shape = (6 * 1000, 1, o, o // 2 + 1)

    print(f"cache={cache_dir}")
    print(f"split={args.split} spectra={spectra.shape} labels={labels.shape}")
    if tuple(spectra.shape) != expected_shape:
        print(f"WARN: expected shape {expected_shape} (6 classes x 1000 samples)")
    if len(spectra) != len(labels):
        raise SystemExit("FAIL: spectra/labels length mismatch")

    # per-class block sanity
    for class_id in range(6):
        block = labels[class_id * 1000 : (class_id + 1) * 1000]
        if not np.all(block == class_id):
            raise SystemExit(f"FAIL: label block for class {class_id} is not contiguous")

    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"metadata seed={meta.get('seed')} samples_per_class={meta.get('samples_per_class_per_size')}")

    print("OK: shared cache ready for Mask and CNN")


if __name__ == "__main__":
    main()
