#!/usr/bin/env python3
"""Plot per-class F1 across methods and input sizes from real metrics files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "comparison" / "per_class_f1"
OUT_PNG = OUT_DIR / "per_class_f1_methods_sizes.png"
OUT_CSV = OUT_DIR / "per_class_f1_methods_sizes.csv"

SIZES = [32, 64, 96, 128]
METHODS = [
    ("Mask", ROOT / "results" / "mask", "n6_mask_size{size}"),
    ("CNN", ROOT / "results" / "cnn", "n6_poscnn_size{size}"),
]
CLASS_ORDER = [
    "original",
    "JPEG_Q80",
    "downsample_x8",
    "downsample_x16",
    "upsample_x4",
    "upsample_x8",
]
CLASS_LABELS = ["Original", "JPEG-Q80", "Down×8", "Down×16", "Up×4", "Up×8"]


def class_f1(metrics: dict, class_name: str) -> float:
    """Handle both Mask and CNN metric JSON layouts."""
    per_class = metrics.get("per_class")
    if isinstance(per_class, dict) and class_name in per_class:
        return float(per_class[class_name]["f1"])

    report = metrics.get("classification_report")
    if isinstance(report, dict) and class_name in report:
        return float(report[class_name]["f1-score"])

    raise KeyError(f"Cannot find F1 for class {class_name!r}")


def load_matrix() -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    columns: list[str] = []
    rows: list[dict[str, object]] = []
    values: list[list[float]] = [[] for _ in CLASS_ORDER]

    for method_name, base_dir, pattern in METHODS:
        for size in SIZES:
            metrics_path = base_dir / pattern.format(size=size) / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)

            metrics = json.loads(metrics_path.read_text())
            columns.append(f"{method_name}\n{size}")

            for class_idx, class_name in enumerate(CLASS_ORDER):
                f1 = class_f1(metrics, class_name)
                values[class_idx].append(f1)
                rows.append(
                    {
                        "method": method_name,
                        "size": size,
                        "class": class_name,
                        "f1": f"{f1:.6f}",
                    }
                )

    return np.array(values, dtype=float), columns, rows


def main() -> None:
    data, columns, rows = load_matrix()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "size", "class", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10.8, 5.6), constrained_layout=True)
    im = ax.imshow(data, cmap="magma", vmin=0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=11)
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_title("Per-class F1 Across Methods and Sizes", fontsize=16, pad=12)

    # Separator between Mask and CNN columns.
    ax.axvline(len(SIZES) - 0.5, color="black", linewidth=2.0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val < 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("F1 score", rotation=90, labelpad=12)
    cbar.set_ticks(np.arange(0.2, 1.01, 0.1))

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
