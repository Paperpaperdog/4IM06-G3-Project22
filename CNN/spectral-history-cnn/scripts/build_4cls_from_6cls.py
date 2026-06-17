#!/usr/bin/env python3
"""Subset 6-class processed spectra cache to 4-class (mask-aligned) labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SOURCE_CLASS_NAMES = [
    "original",
    "JPEG",
    "downsample_x2",
    "downsample_x4",
    "downsample_x8",
    "downsample_x16",
]
TARGET_CLASS_NAMES = ["original", "JPEG", "downsample_x8", "downsample_x16"]
SOURCE_TO_TARGET = {0: 0, 1: 1, 4: 2, 5: 3}


def filter_split(src_dir: Path, dst_dir: Path, split: str) -> None:
    spectra = np.load(src_dir / f"{split}_spectra.npy")
    labels = np.load(src_dir / f"{split}_labels.npy")
    keep = np.isin(labels, list(SOURCE_TO_TARGET.keys()))
    n_total = int(len(labels))
    spectra = spectra[keep]
    labels = labels[keep]
    remapped = np.vectorize(SOURCE_TO_TARGET.get)(labels)

    np.save(dst_dir / f"{split}_spectra.npy", spectra)
    np.save(dst_dir / f"{split}_labels.npy", remapped.astype(np.int64))

    src_meta = src_dir / f"{split}_metadata.csv"
    dst_meta = dst_dir / f"{split}_metadata.csv"
    if src_meta.exists():
        with src_meta.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        kept_rows = [row for row in rows if int(row.get("class_id", -1)) in SOURCE_TO_TARGET]
        for row in kept_rows:
            row["class_id"] = str(SOURCE_TO_TARGET[int(row["class_id"])])
            row["class_name"] = TARGET_CLASS_NAMES[int(row["class_id"])]
        if kept_rows:
            with dst_meta.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()))
                writer.writeheader()
                writer.writerows(kept_rows)

    print(
        f"{split}: kept {len(remapped)}/{n_total} "
        f"spectra, shape={spectra.shape}, labels={np.bincount(remapped)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 4-class cache from 6-class cache.")
    parser.add_argument(
        "--source-dir",
        default="data/processed/v1_final64_tv_rfft",
        help="6-class processed directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/v1_final64_tv_rfft_4cls",
        help="4-class processed directory",
    )
    args = parser.parse_args()

    src_dir = Path(args.source_dir)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        filter_split(src_dir, dst_dir, split)

    with (dst_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(TARGET_CLASS_NAMES, f, indent=2)

    src_cfg = src_dir / "preprocess_config.json"
    if src_cfg.exists():
        meta = json.loads(src_cfg.read_text(encoding="utf-8"))
        meta["derived_from"] = str(src_dir)
        meta["source_class_names"] = SOURCE_CLASS_NAMES
        meta["target_class_names"] = TARGET_CLASS_NAMES
        meta["label_remap"] = {str(k): v for k, v in SOURCE_TO_TARGET.items()}
        (dst_dir / "preprocess_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. 4-class cache -> {dst_dir}")


if __name__ == "__main__":
    main()
