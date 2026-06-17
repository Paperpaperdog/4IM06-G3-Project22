"""Classical NFA detector: input-size sweep with explicit up- and down-sampling.

For each observed (target) size T we synthesize cases in BOTH directions:
  - downsample-to-T: source = T * factor   (image was downsampled from larger)
  - upsample-to-T:   source = T // factor  (image was upsampled from smaller)
run the a-contrario detector on each target, and aggregate, per target size and
per direction, how often the true source size is recovered as the top candidate
(`true_rank == 1`) plus the median significance of the best NFA peak.

This addresses, for the classical route:
  - missing upsampling experiments (we now include source < target);
  - the differing input size across methods (we sweep T explicitly);
  - whether input size affects results (size_effect_summary.csv + plot).

Run (CPU):
  cd 4IM06-G3-Project22
  python scripts/analysis/classical_size_sweep.py \
    --image-dir spectral-mask-resampling/data/raw/raise_tiff \
    --limit-images 20 --target-sizes 256,384,512 --factors 2,4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

# run_controlled_resampling_experiments lives at the project root.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from run_controlled_resampling_experiments import run_experiment  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def source_sizes_for_target(target: int, factors: list[int], min_source: int) -> list[int]:
    """Source sizes producing up- and down-sampling cases to reach `target`."""
    sizes: set[int] = set()
    for f in factors:
        down = target * f  # source larger -> downsample to target
        up = target // f  # source smaller -> upsample to target
        if down >= min_source:
            sizes.add(down)
        if up >= min_source:
            sizes.add(up)
    return sorted(sizes)


def direction(source_size: int, target_size: int) -> str:
    if source_size > target_size:
        return "downsample"
    if source_size < target_size:
        return "upsample"
    return "identity"


def run_one_target(args: argparse.Namespace, target: int, source_sizes: list[int], outdir: Path) -> Path:
    exp_args = Namespace(
        image_dir=args.image_dir,
        raise_csv=args.raise_csv,
        outdir=outdir,
        limit_images=args.limit_images,
        source_sizes=",".join(str(s) for s in source_sizes),
        target_size=target,
        methods=args.methods,
        radius=args.radius,
        tv_weight=args.tv_weight,
        nfa_threshold=args.nfa_threshold,
        top_peak_k=args.top_peak_k,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        tolerance=args.tolerance,
        workers=args.workers,
    )
    return run_experiment(exp_args)


def aggregate(summary_path: Path, target: int) -> list[dict]:
    """Per (target, direction) detection rate + median significance."""
    buckets: dict[str, list[dict]] = {"upsample": [], "downsample": [], "identity": []}
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            src = int(row["source_size"])
            tgt = int(row["target_size"])
            d = direction(src, tgt)
            true_rank = row.get("true_rank", "")
            best_nfa = row.get("best_nfa", "")
            rank_hit = 1 if true_rank not in ("", None) and int(float(true_rank)) == 1 else 0
            try:
                log10_nfa = math.log10(float(best_nfa)) if best_nfa not in ("", None) else float("nan")
            except (ValueError, OverflowError):
                log10_nfa = float("nan")
            buckets[d].append({"rank_hit": rank_hit, "log10_nfa": log10_nfa})

    rows = []
    for d, items in buckets.items():
        if not items:
            continue
        hits = np.array([it["rank_hit"] for it in items], dtype=float)
        nfas = np.array([it["log10_nfa"] for it in items], dtype=float)
        rows.append(
            {
                "target_size": target,
                "direction": d,
                "num_cases": len(items),
                "top1_recovery_rate": float(hits.mean()),
                "median_log10_best_nfa": float(np.nanmedian(nfas)) if len(nfas) else float("nan"),
            }
        )
    return rows


def plot_size_effect(rows: list[dict], out_png: Path) -> None:
    targets = sorted({r["target_size"] for r in rows})
    directions = ["downsample", "upsample"]
    plt.figure(figsize=(8, 5))
    for d in directions:
        ys = []
        for t in targets:
            match = [r for r in rows if r["target_size"] == t and r["direction"] == d]
            ys.append(match[0]["top1_recovery_rate"] if match else float("nan"))
        plt.plot(targets, ys, marker="o", label=d)
    plt.xlabel("Observed (target) size")
    plt.ylabel("Top-1 source-size recovery rate")
    plt.title("Classical NFA: input-size effect (up vs down sampling)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("spectral-mask-resampling/data/raw/raise_tiff"))
    parser.add_argument("--raise-csv", type=Path, default=Path("data/raise_raw/RAISE_1k.csv"))
    parser.add_argument("--outdir-root", type=Path, default=Path("test_results/classical_size_sweep"))
    parser.add_argument("--limit-images", type=int, default=20)
    parser.add_argument("--target-sizes", default="256,384,512")
    parser.add_argument("--factors", default="2,4", help="up/down resampling factors")
    parser.add_argument("--min-source-size", type=int, default=64)
    parser.add_argument("--methods", default="bicubic")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--nfa-threshold", type=float, default=1.0)
    parser.add_argument("--top-peak-k", type=int, default=8)
    parser.add_argument("--min-scale", type=float, default=0.1)
    parser.add_argument("--max-scale", type=float, default=8.0)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers across images per target (0 = all cores).")
    args = parser.parse_args()

    target_sizes = parse_int_list(args.target_sizes)
    factors = parse_int_list(args.factors)
    args.outdir_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for target in target_sizes:
        source_sizes = source_sizes_for_target(target, factors, args.min_source_size)
        if not source_sizes:
            print(f"Skipping target {target}: no valid source sizes.")
            continue
        outdir = args.outdir_root / f"target_{target}"
        print(f"=== target={target} sources={source_sizes} ===")
        summary_path = run_one_target(args, target, source_sizes, outdir)
        all_rows.extend(aggregate(summary_path, target))

    summary_csv = args.outdir_root / "size_effect_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_size", "direction", "num_cases", "top1_recovery_rate", "median_log10_best_nfa"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    plot_size_effect(all_rows, args.outdir_root / "size_effect.png")
    print(f"Saved size-effect summary to {summary_csv}")
    print(f"Saved plot to {args.outdir_root / 'size_effect.png'}")


if __name__ == "__main__":
    main()
