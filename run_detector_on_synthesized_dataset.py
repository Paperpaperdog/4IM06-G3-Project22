"""Run the resampling detector on an already synthesized dataset.

Input is the ``metadata.csv`` produced by
``synthesize_controlled_resampling_dataset.py``. For each target image, the
script writes detector outputs next to ``target.png`` in the existing case
directory.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from candidate_estimation import candidate_rank, rank_size_candidates, significant_distances
from resampling_core import DetectionResult, detect_both_axes, residual_spectrum, tv_residual


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


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_detector(args: argparse.Namespace) -> Path:
    rows = read_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]

    summary_path = args.output_summary
    if summary_path is None:
        summary_path = args.metadata.with_name("detection_summary.csv")

    processed: set[Path] = set()
    with summary_path.open("w", newline="") as summary_handle:
        summary = csv.writer(summary_handle)
        summary.writerow(
            [
                "image_id",
                "source_size",
                "target_size",
                "method",
                "designed_peak",
                "axis",
                "best_distance",
                "best_nfa",
                "true_rank",
                "top_candidate",
                "top_score",
                "target_path",
                "result_dir",
            ]
        )

        for row_index, row in enumerate(rows, start=1):
            target_path = Path(row["target_path"])
            if args.skip_existing and target_path in processed:
                continue

            result_dir = target_path.parent
            if args.skip_existing and (result_dir / "nfa_curves.png").exists():
                continue

            image = np.asarray(Image.open(target_path).convert("L"))
            vertical, horizontal = detect_both_axes(
                image,
                tv_weight=args.tv_weight,
                radius=args.radius,
            )

            save_axis_csv(vertical, result_dir / "vertical_nfa.csv")
            save_axis_csv(horizontal, result_dir / "horizontal_nfa.csv")
            save_peak_csv(vertical, result_dir / "vertical_peaks.csv", args.nfa_threshold, args.top_peak_k)
            save_peak_csv(horizontal, result_dir / "horizontal_peaks.csv", args.nfa_threshold, args.top_peak_k)
            save_nfa_plot(vertical, horizontal, result_dir / "nfa_curves.png", args.nfa_threshold)
            save_spectrum_plot(image, result_dir / "spectrum.png", args.tv_weight)

            source_size = int(row["source_size"])
            target_size = int(row["target_size"])
            for axis_name, result in (("vertical", vertical), ("horizontal", horizontal)):
                candidates = rank_size_candidates(
                    result,
                    target_size,
                    nfa_threshold=args.nfa_threshold,
                    top_peak_k=args.top_peak_k,
                    min_scale=args.min_scale,
                    max_scale=args.max_scale,
                    tolerance=args.tolerance,
                )
                rank = save_candidates_csv(
                    candidates,
                    source_size,
                    result_dir / f"{axis_name}_candidates.csv",
                )
                top = candidates[0] if candidates else None
                summary.writerow(
                    [
                        row.get("image_id", ""),
                        source_size,
                        target_size,
                        row.get("method", ""),
                        row.get("designed_peak", ""),
                        axis_name,
                        result.best_distance,
                        f"{result.best_nfa:.6g}",
                        "" if rank is None else rank,
                        "" if top is None else top.size,
                        "" if top is None else f"{top.score:.6g}",
                        target_path,
                        result_dir,
                    ]
                )

            processed.add(target_path)
            if args.progress_every and row_index % args.progress_every == 0:
                print(f"Processed {row_index} metadata rows")

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata",
        type=Path,
        nargs="?",
        default=Path("test_results/controlled_resampling_dataset_bicubic_raise100/metadata.csv"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument("--nfa-threshold", type=float, default=1.0)
    parser.add_argument("--top-peak-k", type=int, default=8)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    summary_path = run_detector(args)
    print(f"Saved detection summary to: {summary_path}")


if __name__ == "__main__":
    main()
