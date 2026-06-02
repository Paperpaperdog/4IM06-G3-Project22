from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import (
    GENERATED_DIR,
    IDEA1_PROBE_DISTANCES,
    K_VALUES,
    PILOT_RESULTS_DIR,
    REFERENCE_SIZES,
    TARGET_SIZE,
)
from .io_utils import list_png_subset, load_grayscale, read_manifest, save_grayscale_png
from .metrics import detection_summary
from .transforms import apply_jpeg, reference_sizes_for_group, resize_gray, resample_via_reference


PATTERNS = ("ref_plus_k", "target_minus_ref_plus_k")


def run_idea2(
    png_paths: list[Path],
    generated_dir: Path,
    results_csv: Path,
    jpeg_quality: int | None,
) -> None:
    rows: list[dict] = []

    for png_path in png_paths:
        square = load_grayscale(png_path)
        image_id = png_path.stem
        source_side = square.shape[0]

        for reference_peak in REFERENCE_SIZES:
            for pattern in PATTERNS:
                for k in K_VALUES:
                    ref_sizes = reference_sizes_for_group(
                        reference_peak, k, TARGET_SIZE, pattern
                    )
                    for variant_index, reference_size in enumerate(ref_sizes):
                        transformed, _ = resample_via_reference(
                            square, reference_size, TARGET_SIZE
                        )
                        if jpeg_quality is not None:
                            transformed = apply_jpeg(transformed, jpeg_quality)

                        rel_dir = (
                            generated_dir
                            / "idea2"
                            / image_id
                            / f"ref{reference_peak}"
                            / pattern
                            / f"k{k}"
                        )
                        rel_dir.mkdir(parents=True, exist_ok=True)
                        out_path = rel_dir / f"v{variant_index}_r{reference_size}.png"
                        save_grayscale_png(out_path, transformed)

                        summary = detection_summary(transformed, IDEA1_PROBE_DISTANCES)
                        row = {
                            "pilot": "idea2",
                            "image_id": image_id,
                            "source_side": source_side,
                            "reference_peak": reference_peak,
                            "pattern": pattern,
                            "k": k,
                            "variant_index": variant_index,
                            "reference_size": reference_size,
                            "target_size": TARGET_SIZE,
                            "expected_period_mod": abs(reference_size - TARGET_SIZE)
                            % TARGET_SIZE,
                            "output_path": str(out_path),
                            **summary,
                        }
                        rows.append(row)

        print(f"idea2: finished {image_id} ({len(rows)} rows so far)")

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Idea 2: {len(rows)} rows -> {results_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Idea 2: k in {-1,0,1} + shape metrics.")
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--results-csv", type=Path, default=PILOT_RESULTS_DIR / "idea2_results.csv")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    args = parser.parse_args()

    manifest = read_manifest()
    if manifest:
        png_paths = [Path(row["png_path"]) for row in manifest]
    else:
        png_paths = list_png_subset()
    if not png_paths:
        raise SystemExit("No PNG subset. Run: python -m pilots.prepare_subset")

    run_idea2(png_paths, args.generated_dir, args.results_csv, args.jpeg_quality)


if __name__ == "__main__":
    main()
