"""Aggregate the unified-class input-size sweep across Mask and CNN.

Reads the per-size ``metrics.json`` produced by each method's evaluate step and
produces a single comparison table + plot of test accuracy vs observed input
size. This is the direct answer to "does input size affect results?", now that
both learnable methods share the same 6 classes (incl. upsampling).

Run:
  cd 4IM06-G3-Project22
  python scripts/analysis/summarize_size_effect.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def method_paths(variant: str) -> dict[str, Path]:
    if variant not in ("u6", "u7", "n6"):
        raise ValueError(f"variant must be u6, u7 or n6, got {variant!r}")
    return {
        "mask": PROJECT_ROOT / "spectral-mask-resampling" / "outputs" / f"{variant}_mask_size{{size}}" / "metrics.json",
        "cnn": PROJECT_ROOT / "CNN" / "spectral-history-cnn" / "outputs" / f"{variant}_poscnn_size{{size}}" / "metrics.json",
    }


def macro_f1(metrics: dict) -> float | None:
    """Macro F1 across both metrics.json layouts (mask vs CNN)."""
    report = metrics.get("classification_report")
    if isinstance(report, dict):
        f1s = [v["f1-score"] for k, v in report.items() if isinstance(v, dict) and "f1-score" in v
               and k not in ("accuracy", "macro avg", "weighted avg")]
        if f1s:
            return sum(f1s) / len(f1s)
    per_class = metrics.get("per_class")
    if isinstance(per_class, dict):
        f1s = [v["f1"] for v in per_class.values() if isinstance(v, dict) and "f1" in v]
        if f1s:
            return sum(f1s) / len(f1s)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="32,64,96,128")
    parser.add_argument("--variant", default="u6", choices=["u6", "u7", "n6"],
                        help="u6=6-class main sweep; u7=adds upsample_x8 (7 classes); "
                             "n6=native-spectrum 6-class set (ds8/ds16/up4/up8).")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    methods = method_paths(args.variant)
    _suffix = {"u7": "_u7", "n6": "_n6"}.get(args.variant, "")
    args.outdir = args.outdir or (PROJECT_ROOT / "test_results" / f"size_effect{_suffix}")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    series: dict[str, dict[int, float]] = {m: {} for m in methods}
    for method, template in methods.items():
        for size in sizes:
            path = Path(str(template).format(size=size))
            if not path.exists():
                print(f"[skip] {method} size={size}: missing {path}")
                continue
            metrics = json.loads(path.read_text())
            acc = float(metrics.get("accuracy", float("nan")))
            series[method][size] = acc
            rows.append(
                {
                    "method": method,
                    "observed_size": size,
                    "accuracy": acc,
                    "macro_f1": macro_f1(metrics),
                }
            )

    csv_path = args.outdir / "size_effect_combined.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "observed_size", "accuracy", "macro_f1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    plt.figure(figsize=(8, 5))
    for method, by_size in series.items():
        if not by_size:
            continue
        xs = sorted(by_size)
        ys = [by_size[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Observed input size")
    plt.ylabel("Test accuracy")
    plt.title("Input-size effect (unified 6 classes, incl. upsampling)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_path = args.outdir / "size_effect.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    if not rows:
        print("No metrics found yet. Run the size sweeps first.")


if __name__ == "__main__":
    main()
