"""Small demo for the minimal spectral-correlation resampling detector.

Usage:
    python3 demo_resampling_detection.py
    python3 demo_resampling_detection.py path/to/image.png

With no image path, the script creates a synthetic resampling example from
``skimage.data.camera``. It prints the distances with the smallest NFAs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data
from skimage.transform import resize

from resampling_core import DetectionResult, detect_both_axes


def load_image(path: str | None) -> np.ndarray:
    if path is None:
        original = data.camera()
        return resize(
            original,
            (384, 384),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.uint8)

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def print_top_distances(result: DetectionResult, top_k: int = 8) -> None:
    order = np.argsort(result.nfa)[:top_k]
    axis_name = "vertical/rows" if result.axis == 0 else "horizontal/cols"
    print(f"\nAxis: {axis_name}")
    print("distance  maxima_count  NFA          log10(NFA)")
    for index in order:
        print(
            f"{int(result.distances[index]):8d}  "
            f"{int(result.maxima_counts[index]):12d}  "
            f"{result.nfa[index]:.3e}  "
            f"{result.log10_nfa[index]:10.3f}"
        )


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


def save_nfa_plot(
    vertical: DetectionResult,
    horizontal: DetectionResult,
    output_path: Path,
    threshold: float | None = 1.0,
) -> None:
    plt.figure(figsize=(11, 5))
    plt.semilogy(vertical.distances, vertical.nfa, label="Vertical / rows", linewidth=1.5)
    plt.semilogy(horizontal.distances, horizontal.nfa, label="Horizontal / cols", linewidth=1.5)

    if threshold is not None:
        plt.axhline(
            threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"NFA threshold = {threshold:g}",
        )

    plt.scatter(
        [vertical.best_distance],
        [vertical.best_nfa],
        color="tab:blue",
        s=35,
        zorder=3,
    )
    plt.scatter(
        [horizontal.best_distance],
        [horizontal.best_nfa],
        color="tab:orange",
        s=35,
        zorder=3,
    )
    plt.xlabel("Tested distance d")
    plt.ylabel("NFA")
    plt.title("Resampling Detection: NFA for All Tested Distances")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_test_results(
    vertical: DetectionResult,
    horizontal: DetectionResult,
    image_path: str | None,
    outdir: Path,
    threshold: float | None,
) -> Path:
    image_name = Path(image_path).stem if image_path else "skimage_camera_512_to_384"
    result_dir = outdir / image_name
    result_dir.mkdir(parents=True, exist_ok=True)

    save_axis_csv(vertical, result_dir / "vertical_nfa.csv")
    save_axis_csv(horizontal, result_dir / "horizontal_nfa.csv")
    save_nfa_plot(vertical, horizontal, result_dir / "nfa_all_distances.png", threshold)
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="optional image path")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=1.0)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("test_results/resampling_detection"),
        help="directory where plots and CSV results are saved",
    )
    parser.add_argument(
        "--nfa-threshold",
        type=float,
        default=1.0,
        help="optional horizontal threshold shown in the NFA plot",
    )
    args = parser.parse_args()

    image = load_image(args.image)
    vertical, horizontal = detect_both_axes(
        image,
        tv_weight=args.tv_weight,
        radius=args.radius,
    )

    print(f"Image shape: {image.shape}")
    print(f"Patch shape: {vertical.patch_shape}")
    print_top_distances(vertical)
    print_top_distances(horizontal)

    result_dir = save_test_results(
        vertical,
        horizontal,
        image_path=args.image,
        outdir=args.outdir,
        threshold=args.nfa_threshold,
    )
    print(f"\nSaved test results to: {result_dir}")
    print(f"NFA plot: {result_dir / 'nfa_all_distances.png'}")
    print(f"NFA values: {result_dir / 'vertical_nfa.csv'}")
    print(f"NFA values: {result_dir / 'horizontal_nfa.csv'}")


if __name__ == "__main__":
    main()
