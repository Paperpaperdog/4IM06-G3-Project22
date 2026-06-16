from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import (
    GENERATED_DIR,
    IDEA1_PROBE_DISTANCES,
    PILOT_RESULTS_DIR,
    TARGET_SIZE,
)
from .io_utils import list_png_subset, load_grayscale, read_manifest, save_grayscale_png
from .metrics import detection_summary, jpeg_period_distances
from .transforms import apply_jpeg, resize_gray, resample_via_reference, simulate_x8_block_resample


def idea1_conditions(source_side: int) -> list[dict]:
    """Controlled variants for JPEG vs resampling confusion."""
    return [
        {
            "condition": "png_identity",
            "jpeg_q": None,
            "transform": "identity",
            "expected_period": 0,
        },
        {
            "condition": "jpeg_q90_identity",
            "jpeg_q": 90,
            "transform": "identity",
            "expected_period": 0,
        },
        {
            "condition": "jpeg_q75_identity",
            "jpeg_q": 75,
            "transform": "identity",
            "expected_period": 0,
        },
        {
            "condition": "png_resample_to_target",
            "jpeg_q": None,
            "transform": "resample_to_target",
            "expected_period": abs(source_side - TARGET_SIZE) if source_side != TARGET_SIZE else 128,
        },
        {
            "condition": "jpeg_q90_resample_to_target",
            "jpeg_q": 90,
            "transform": "resample_to_target",
            "expected_period": abs(source_side - TARGET_SIZE) if source_side != TARGET_SIZE else 128,
        },
        {
            "condition": "png_sim_x8",
            "jpeg_q": None,
            "transform": "sim_x8",
            "expected_period": TARGET_SIZE // 8,
        },
        {
            "condition": "jpeg_q90_sim_x8",
            "jpeg_q": 90,
            "transform": "sim_x8",
            "expected_period": TARGET_SIZE // 8,
        },
    ]


def apply_condition(square: np.ndarray, spec: dict) -> np.ndarray:
    transform = spec["transform"]
    if transform == "identity":
        out = resize_gray(square, TARGET_SIZE)
    elif transform == "resample_to_target":
        out = resize_gray(square, TARGET_SIZE)
    elif transform == "sim_x8":
        base = resize_gray(square, TARGET_SIZE)
        out = simulate_x8_block_resample(base, TARGET_SIZE)
    else:
        raise ValueError(transform)

    if spec["jpeg_q"] is not None:
        out = apply_jpeg(out, int(spec["jpeg_q"]))
    return out


def run_idea1(
    png_paths: list[Path],
    generated_dir: Path,
    results_csv: Path,
) -> None:
    rows: list[dict] = []
    jpeg_periods = jpeg_period_distances(TARGET_SIZE)

    for png_path in png_paths:
        square = load_grayscale(png_path)
        image_id = png_path.stem
        for spec in idea1_conditions(square.shape[0]):
            transformed = apply_condition(square, spec)
            rel_dir = generated_dir / "idea1" / image_id
            rel_dir.mkdir(parents=True, exist_ok=True)
            out_path = rel_dir / f"{spec['condition']}.png"
            save_grayscale_png(out_path, transformed)

            summary = detection_summary(transformed, IDEA1_PROBE_DISTANCES)
            row = {
                "pilot": "idea1",
                "image_id": image_id,
                "condition": spec["condition"],
                "transform": spec["transform"],
                "jpeg_q": spec["jpeg_q"] if spec["jpeg_q"] is not None else "",
                "expected_period": spec["expected_period"],
                "output_path": str(out_path),
                **summary,
            }
            for period in jpeg_periods:
                row[f"is_jpeg_period_{period}"] = int(period in IDEA1_PROBE_DISTANCES)
            rows.append(row)
            print(f"idea1: {image_id} / {spec['condition']} -> best d={int(summary['nfa_best_distance'])}")

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Idea 1: {len(rows)} rows -> {results_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Idea 1: JPEG / x8 / x16 confusion.")
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--results-csv", type=Path, default=PILOT_RESULTS_DIR / "idea1_results.csv")
    args = parser.parse_args()

    manifest = read_manifest()
    if manifest:
        png_paths = [Path(row["png_path"]) for row in manifest]
    else:
        png_paths = list_png_subset()
    if not png_paths:
        raise SystemExit("No PNG subset. Run: python -m pilots.prepare_subset")

    run_idea1(png_paths, args.generated_dir, args.results_csv)


if __name__ == "__main__":
    main()
