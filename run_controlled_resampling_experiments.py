"""Controlled experiments for original-size candidate estimation.

Pipeline for each TIFF image:
1. center-crop to a square and save a PNG reference;
2. resize the PNG reference to a controlled source size;
3. resize that controlled source to a fixed target size with a chosen method;
4. run the spectral resampling detector;
5. save spectrum, NFA curves, peaks, candidates, and summary CSV files.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from candidate_estimation import candidate_rank, rank_size_candidates, significant_distances
from resampling_core import DetectionResult, detect_both_axes, residual_spectrum, tv_residual


RESAMPLE_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def image_id(path: Path) -> str:
    return path.stem.lower()


def discover_tiff_paths(image_dir: Path, csv_path: Path | None, limit: int | None) -> list[Path]:
    """Find local TIFF files, optionally ordered by a RAISE metadata CSV."""

    by_id: dict[str, Path] = {}
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for path in image_dir.glob(pattern):
            by_id[image_id(path)] = path

    if csv_path is None or not csv_path.is_file():
        if csv_path is not None and not csv_path.is_file():
            print(f"[Warn] RAISE CSV not found ({csv_path}); using sorted TIFF filenames.")
        paths = [by_id[key] for key in sorted(by_id)]
    else:
        paths = []
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = row.get("File", "").lower()
                if key in by_id:
                    paths.append(by_id[key])

    if limit is not None:
        paths = paths[:limit]
    return paths


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def load_reference_png(tiff_path: Path, output_dir: Path) -> Image.Image:
    """Convert the TIFF to an RGB square PNG reference and return it."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{tiff_path.stem}_reference.png"
    with Image.open(tiff_path) as image:
        reference = center_crop_square(image.convert("RGB"))
    reference.save(png_path)
    return reference


def resize_square(image: Image.Image, size: int, method: str) -> Image.Image:
    return image.resize((size, size), RESAMPLE_FILTERS[method])


def save_axis_csv(result: DetectionResult, path: Path) -> None:
    table = np.column_stack(
        [
            result.distances,
            result.maxima_counts,
            result.nfa,
            result.log10_nfa,
        ]
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="distance,maxima_count,nfa,log10_nfa",
        comments="",
    )


def save_candidates_csv(candidates, true_size: int, path: Path) -> int | None:
    rank = candidate_rank(candidates, true_size)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "candidate_size",
                "score",
                "is_true_size",
                "predicted_distances",
                "supporting_distances",
                "support_scores",
            ]
        )
        for index, candidate in enumerate(candidates, start=1):
            writer.writerow(
                [
                    index,
                    candidate.size,
                    f"{candidate.score:.6g}",
                    int(candidate.size == true_size),
                    " ".join(map(str, candidate.predicted_distances)),
                    " ".join(map(str, candidate.supporting_distances)),
                    " ".join(f"{score:.6g}" for score in candidate.support_scores),
                ]
            )
    return rank


def save_peak_csv(result: DetectionResult, path: Path, nfa_threshold: float | None, top_k: int) -> None:
    peaks = significant_distances(result, nfa_threshold=nfa_threshold, top_k=top_k)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance", "maxima_count", "nfa", "log10_nfa"])
        for distance in peaks:
            index = int(np.where(result.distances == distance)[0][0])
            writer.writerow(
                [
                    int(distance),
                    int(result.maxima_counts[index]),
                    f"{result.nfa[index]:.6g}",
                    f"{result.log10_nfa[index]:.6g}",
                ]
            )


def save_nfa_plot(
    vertical: DetectionResult,
    horizontal: DetectionResult,
    output_path: Path,
    threshold: float | None,
) -> None:
    plt.figure(figsize=(11, 5))
    plt.semilogy(vertical.distances, vertical.nfa, label="Vertical / rows", linewidth=1.5)
    plt.semilogy(horizontal.distances, horizontal.nfa, label="Horizontal / cols", linewidth=1.5)
    if threshold is not None:
        plt.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.0)
    plt.scatter([vertical.best_distance], [vertical.best_nfa], color="tab:blue", s=35)
    plt.scatter([horizontal.best_distance], [horizontal.best_nfa], color="tab:orange", s=35)
    plt.xlabel("Tested distance d")
    plt.ylabel("NFA")
    plt.title("NFA curves")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_spectrum_plot(image: np.ndarray, output_path: Path, tv_weight: float) -> None:
    residual = tv_residual(image, weight=tv_weight)
    spectrum = residual_spectrum(residual)
    magnitude = np.log1p(np.abs(spectrum))
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap="magma")
    plt.axis("off")
    plt.title("Log residual spectrum magnitude")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=160)
    plt.close()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


SUMMARY_HEADER = [
    "image_id",
    "source_size",
    "target_size",
    "method",
    "axis",
    "best_distance",
    "best_nfa",
    "true_rank",
    "top_candidate",
    "top_score",
    "result_dir",
]


def process_image(tiff_path: Path, args: argparse.Namespace) -> list[list]:
    """Run every (source_size, method, axis) case for one image.

    Each image writes only to its own per-id subdirectories, so images can be
    processed independently in parallel. Returns the summary rows for this image.
    """
    source_sizes = parse_int_list(args.source_sizes)
    methods = parse_str_list(args.methods)

    rows: list[list] = []
    reference = load_reference_png(tiff_path, args.outdir / "references")
    for source_size in source_sizes:
        source = resize_square(reference, source_size, "lanczos")
        source_dir = args.outdir / "controlled_sources" / image_id(tiff_path)
        source_dir.mkdir(parents=True, exist_ok=True)
        source_png = source_dir / f"source_{source_size}.png"
        source.save(source_png)

        for method in methods:
            target = resize_square(source, args.target_size, method)
            target_array = np.asarray(target.convert("L"))

            case_dir = (
                args.outdir
                / "cases"
                / image_id(tiff_path)
                / f"source_{source_size}_target_{args.target_size}_{method}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            target.save(case_dir / "target.png")

            vertical, horizontal = detect_both_axes(
                target_array,
                tv_weight=args.tv_weight,
                radius=args.radius,
            )

            save_axis_csv(vertical, case_dir / "vertical_nfa.csv")
            save_axis_csv(horizontal, case_dir / "horizontal_nfa.csv")
            save_peak_csv(
                vertical,
                case_dir / "vertical_peaks.csv",
                args.nfa_threshold,
                args.top_peak_k,
            )
            save_peak_csv(
                horizontal,
                case_dir / "horizontal_peaks.csv",
                args.nfa_threshold,
                args.top_peak_k,
            )
            save_nfa_plot(vertical, horizontal, case_dir / "nfa_curves.png", args.nfa_threshold)
            save_spectrum_plot(target_array, case_dir / "spectrum.png", args.tv_weight)

            for axis_name, result in (("vertical", vertical), ("horizontal", horizontal)):
                candidates = rank_size_candidates(
                    result,
                    args.target_size,
                    nfa_threshold=args.nfa_threshold,
                    top_peak_k=args.top_peak_k,
                    min_scale=args.min_scale,
                    max_scale=args.max_scale,
                    tolerance=args.tolerance,
                )
                rank = save_candidates_csv(
                    candidates,
                    source_size,
                    case_dir / f"{axis_name}_candidates.csv",
                )
                top = candidates[0] if candidates else None
                rows.append(
                    [
                        image_id(tiff_path),
                        source_size,
                        args.target_size,
                        method,
                        axis_name,
                        result.best_distance,
                        f"{result.best_nfa:.6g}",
                        "" if rank is None else rank,
                        "" if top is None else top.size,
                        "" if top is None else f"{top.score:.6g}",
                        case_dir,
                    ]
                )
    return rows


def _process_image_star(packed):
    tiff_path, args = packed
    return process_image(tiff_path, args)


def run_experiment(args: argparse.Namespace) -> Path:
    tiff_paths = discover_tiff_paths(args.image_dir, args.raise_csv, args.limit_images)
    if not tiff_paths:
        raise RuntimeError(f"no TIFF images found in {args.image_dir}")

    methods = parse_str_list(args.methods)
    unknown_methods = sorted(set(methods) - set(RESAMPLE_FILTERS))
    if unknown_methods:
        raise ValueError(f"unknown interpolation methods: {', '.join(unknown_methods)}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    workers = (os.cpu_count() or 1) if args.workers == 0 else max(1, args.workers)
    workers = min(workers, len(tiff_paths))

    # pool.map preserves task order, so the summary stays identical to the
    # original sequential run regardless of how many workers are used.
    if workers == 1:
        per_image_rows = [process_image(path, args) for path in tiff_paths]
    else:
        print(f"[Info] Processing {len(tiff_paths)} images across {workers} workers ...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per_image_rows = list(
                pool.map(_process_image_star, [(path, args) for path in tiff_paths])
            )

    summary_path = args.outdir / "summary.csv"
    with summary_path.open("w", newline="") as summary_handle:
        summary = csv.writer(summary_handle)
        summary.writerow(SUMMARY_HEADER)
        for rows in per_image_rows:
            for row in rows:
                summary.writerow(row)

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("test_images"))
    parser.add_argument("--raise-csv", type=Path, default=Path("data/raise_raw/RAISE_1k.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("test_results/controlled_resampling"))
    parser.add_argument("--limit-images", type=int, default=1)
    parser.add_argument("--source-sizes", default="256,320,448,512,640")
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--methods", default="nearest,bilinear,bicubic,lanczos")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--nfa-threshold", type=float, default=1.0)
    parser.add_argument("--top-peak-k", type=int, default=8)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes across images. 0 = all CPU cores, 1 = single process.",
    )
    args = parser.parse_args()

    summary_path = run_experiment(args)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
