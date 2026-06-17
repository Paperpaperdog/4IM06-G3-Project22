"""Input-size sweep for the DCT-FFT a-contrario JPEG-vs-resampling detector.

Runs evaluate_detector_on_dataset.py at several `--max_size` values over a
forensic post-processing dataset (build it with create_forensic_postprocess_dataset.py
--include_upsampling so up-sampling classes are present), and aggregates overall
accuracy vs input size. This is route A's DCT-FFT view, complementary to the NFA
controlled experiment (classical_size_sweep.py).

Note: the detector has no native "upsampling" label, so up-sampling rows mostly
fall into original/resampling — that confusion is itself the reported result.

Run (CPU):
  cd 4IM06-G3-Project22
  # 1. build dataset with upsampling
  python create_forensic_postprocess_dataset.py \
    --input_dir <png_dir> --output_dir test_results/forensic_pp --include_original \
    --include_upsampling --mix_order both
  # 2. sweep input size
  python scripts/analysis/jpeg_detector_size_sweep.py \
    --dataset-root test_results/forensic_pp \
    --null-dir test_results/forensic_pp/original
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACC_RE = re.compile(r"Accuracy:\s*([0-9.]+)")
BIN_RE = re.compile(r"BinaryResamplingAccuracy:\s*([0-9.]+)")


def run_one(detector: Path, evaluator: Path, dataset_root: Path, null_dir: Path,
            max_size: int, max_per_class: int, workers: int,
            json_out: Path) -> tuple[float | None, float | None]:
    cmd = [
        sys.executable, str(evaluator),
        "--detector", str(detector),
        "--dataset_root", str(dataset_root),
        "--split", "none",
        "--null_dir", str(null_dir),
        "--max_size", str(max_size),
        "--max_per_class", str(max_per_class),
        "--workers", str(workers),
        "--json_out", str(json_out),
    ]
    print(f"=== max_size={max_size} ===")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout[-2000:])
    acc_m = ACC_RE.findall(proc.stdout)
    bin_m = BIN_RE.findall(proc.stdout)
    acc = float(acc_m[-1]) if acc_m else None
    binacc = float(bin_m[-1]) if bin_m else None
    return acc, binacc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, default=PROJECT_ROOT / "jpeg_resample_detector.py")
    parser.add_argument("--evaluator", type=Path, default=PROJECT_ROOT / "evaluate_detector_on_dataset.py")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--null-dir", type=Path, required=True)
    parser.add_argument("--max-sizes", default="128,256,512")
    parser.add_argument("--max-per-class", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers passed to the evaluator (0 = all cores).")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "test_results" / "jpeg_detector_size_sweep")
    args = parser.parse_args()

    max_sizes = [int(s) for s in args.max_sizes.split(",") if s.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for max_size in max_sizes:
        json_out = args.outdir / f"eval_size{max_size}.json"
        acc, binacc = run_one(args.detector, args.evaluator, args.dataset_root, args.null_dir,
                              max_size, args.max_per_class, args.workers, json_out)
        rows.append({"max_size": max_size, "accuracy": acc, "binary_resampling_accuracy": binacc})

    csv_path = args.outdir / "size_effect_jpeg_detector.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["max_size", "accuracy", "binary_resampling_accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    xs = [r["max_size"] for r in rows if r["accuracy"] is not None]
    ys = [r["accuracy"] for r in rows if r["accuracy"] is not None]
    if xs:
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("max_size (input size)")
        plt.ylabel("Detector accuracy")
        plt.title("JPEG-vs-resampling detector: input-size effect")
        plt.ylim(0, 1)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(args.outdir / "size_effect.png", dpi=160)
        plt.close()

    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
