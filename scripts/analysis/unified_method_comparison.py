"""Unified three-method comparison on one common axis.

The three routes do *different* native tasks (classical NFA = period-8 / source-size
recovery; Mask & CNN = 6-way classification), so they are not directly comparable
on a single 6-class accuracy. This script collapses every method onto the largest
fair common ground:

    binary task = "was the image geometrically resampled (up/down)?"  vs  "not"
                  (original and JPEG-only count as NOT resampled).

For each input size it reports each method's binary "resampling-detected" accuracy:
  - Mask / CNN: collapse the saved 6x6 confusion_matrix (per-size metrics.json) into
    {resampled = downsample_*/upsample_*}  vs  {not = original/JPEG}.
  - Classical (route A, DCT-FFT detector): read the per-size eval JSON written by
    jpeg_detector_size_sweep.py (its `binary_resampling_accuracy`).

Output: one CSV + one figure with all three methods, plus the 6-class accuracy of
Mask/CNN for context.

Run (after the size sweeps have produced their outputs):
  cd 4IM06-G3-Project22
  python scripts/analysis/unified_method_comparison.py --sizes 32,64,96,128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def class_names_from_config(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text())
    if isinstance(cfg.get("class_names"), list):
        return list(cfg["class_names"])
    data = cfg.get("data", {})
    if isinstance(data.get("classes"), list):
        return list(data["classes"])
    raise KeyError(f"Could not find class names in {config_path}")


def is_resampling_class(name: str) -> bool:
    return name.startswith("downsample_") or name.startswith("upsample_")


def binary_accuracy_from_confusion(confusion: list[list[float]], class_names: list[str]) -> float:
    """Collapse a 6x6 (true x pred) confusion matrix into resampled-vs-not accuracy."""
    res = [is_resampling_class(n) for n in class_names]
    total = 0.0
    correct = 0.0
    for i, row in enumerate(confusion):
        for j, count in enumerate(row):
            total += count
            if res[i] == res[j]:
                correct += count
    return correct / total if total > 0 else float("nan")


LEARNABLE = {
    "mask": {
        "metrics": PROJECT_ROOT / "spectral-mask-resampling" / "outputs" / "u6_mask_size{size}" / "metrics.json",
        "config": PROJECT_ROOT / "spectral-mask-resampling" / "configs" / "size_sweep" / "u6_mask_size{size}.yaml",
    },
    "cnn": {
        "metrics": PROJECT_ROOT / "CNN" / "spectral-history-cnn" / "outputs" / "u6_poscnn_size{size}" / "metrics.json",
        "config": PROJECT_ROOT / "CNN" / "spectral-history-cnn" / "configs" / "size_sweep" / "u6_poscnn_size{size}.yaml",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="32,64,96,128")
    parser.add_argument(
        "--classical-eval-dir",
        type=Path,
        default=PROJECT_ROOT / "test_results" / "jpeg_detector_size_sweep",
        help="Dir with eval_size{size}.json from jpeg_detector_size_sweep.py "
             "(run it with --max-sizes equal to --sizes here).",
    )
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "test_results" / "unified_comparison")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    series: dict[str, dict[int, float]] = {"mask": {}, "cnn": {}, "classical": {}}

    # Mask + CNN: collapse 6-class confusion to the binary axis.
    for method, paths in LEARNABLE.items():
        for size in sizes:
            metrics_path = Path(str(paths["metrics"]).format(size=size))
            config_path = Path(str(paths["config"]).format(size=size))
            if not metrics_path.exists():
                print(f"[skip] {method} size={size}: missing {metrics_path}")
                continue
            metrics = json.loads(metrics_path.read_text())
            confusion = metrics.get("confusion_matrix")
            class_names = class_names_from_config(config_path)
            binacc = binary_accuracy_from_confusion(confusion, class_names) if confusion else float("nan")
            series[method][size] = binacc
            rows.append({
                "method": method,
                "input_size": size,
                "binary_resampling_accuracy": binacc,
                "six_class_accuracy": float(metrics.get("accuracy", float("nan"))),
            })

    # Classical route A (DCT-FFT detector): read its per-size binary accuracy.
    for size in sizes:
        eval_path = args.classical_eval_dir / f"eval_size{size}.json"
        if not eval_path.exists():
            print(f"[skip] classical size={size}: missing {eval_path}")
            continue
        data = json.loads(eval_path.read_text())
        binacc = float(data.get("binary_resampling_accuracy", float("nan")))
        series["classical"][size] = binacc
        rows.append({
            "method": "classical",
            "input_size": size,
            "binary_resampling_accuracy": binacc,
            "six_class_accuracy": "",  # not a 6-class classifier by design
        })

    csv_path = args.outdir / "unified_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "input_size", "binary_resampling_accuracy", "six_class_accuracy"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["method"], r["input_size"])):
            writer.writerow(row)

    plt.figure(figsize=(8, 5))
    for method in ("classical", "mask", "cnn"):
        by_size = series[method]
        if not by_size:
            continue
        xs = sorted(by_size)
        ys = [by_size[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Input size (px)")
    plt.ylabel("Binary 'resampling detected' accuracy")
    plt.title("Three methods on the common binary axis (resampled vs original)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_path = args.outdir / "unified_comparison.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    if not rows:
        print("No metrics found yet. Run the three size sweeps first.")


if __name__ == "__main__":
    main()
