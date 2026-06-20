"""Per-class F1 heatmaps for Mask and CNN across n6 size sweep.

Reads results/comparison/raw_metrics_all.csv (or metrics.json per run) and writes:
  results/comparison/per_class_f1_heatmap.png
  results/comparison/per_class_f1_heatmap.csv

Run:
  cd 4IM06-G3-Project22
  python scripts/analysis/plot_per_class_f1_heatmap.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "results" / "comparison" / "raw_metrics_all.csv"
OUT_DIR = PROJECT_ROOT / "results" / "comparison"

CLASS_NAMES = [
    "original",
    "JPEG_Q80",
    "downsample_x8",
    "downsample_x16",
    "upsample_x4",
    "upsample_x8",
]
METHODS = ["Mask", "CNN"]
SIZES = [32, 64, 96, 128]


def load_f1_table() -> dict[tuple[str, int], dict[str, float]]:
    table: dict[tuple[str, int], dict[str, float]] = {}
    with RAW_CSV.open() as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            size = int(row["size"])
            f1s = {}
            for name in CLASS_NAMES:
                key = f"{name}|f1-score"
                f1s[name] = float(row[key])
            table[(method, size)] = f1s
    return table


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_CSV}; run export first.")

    table = load_f1_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Long CSV for reports
    csv_path = OUT_DIR / "per_class_f1_heatmap.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "size", *CLASS_NAMES])
        writer.writeheader()
        for method in METHODS:
            for size in SIZES:
                f1s = table.get((method, size))
                if not f1s:
                    continue
                writer.writerow({"method": method, "size": size, **f1s})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, method in zip(axes, METHODS):
        mat = np.array(
            [[table[(method, size)][c] for c in CLASS_NAMES] for size in SIZES],
            dtype=float,
        )
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            xticklabels=CLASS_NAMES,
            yticklabels=[str(s) for s in SIZES],
            ax=ax,
            cbar=ax is axes[-1],
            cbar_kws={"label": "F1"},
        )
        ax.set_title(method)
        ax.set_xlabel("Class")
        ax.set_ylabel("Observed size (px)")

    fig.suptitle("Per-class F1 across methods & sizes (n6 test)", y=1.02, fontsize=13)
    fig.tight_layout()
    png_path = OUT_DIR / "per_class_f1_heatmap.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
