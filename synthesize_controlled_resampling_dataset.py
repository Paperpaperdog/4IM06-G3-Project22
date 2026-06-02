"""Batch synthesize controlled resampling data from local RAISE TIFF images.

This script only creates images and metadata. It does not run the detector.

Default design:
- target sizes: 256, 384, 512
- three designed residues/peaks per target size
- bicubic final interpolation
- five source/reference sizes per (target, peak) group

Directory layout follows the controlled experiment script:

    outdir/
      references/<image>_reference.png
      controlled_sources/<image>/source_<N>.png
      cases/<image>/source_<N>_target_<C>_bicubic/target.png
      metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


RESAMPLE_FILTERS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


DEFAULT_PEAKS = {
    256: (64, 85, 96),
    384: (96, 128, 160),
    512: (128, 171, 192),
}


@dataclass(frozen=True)
class SynthesisCase:
    target_size: int
    designed_peak: int
    source_size: int
    residue: int
    residue_family: str


def image_id(path: Path) -> str:
    return path.stem.lower()


def discover_tiff_paths(image_dir: Path, csv_path: Path | None, limit: int | None) -> list[Path]:
    """Find local TIFF files, optionally ordered by a RAISE metadata CSV."""

    by_id: dict[str, Path] = {}
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for path in image_dir.glob(pattern):
            by_id[image_id(path)] = path

    if csv_path is None:
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


def download_tiff_paths(
    csv_path: Path,
    download_dir: Path,
    limit: int,
    *,
    force_download: bool = False,
) -> list[Path]:
    """Download the first ``limit`` TIFF files listed in a RAISE metadata CSV."""

    download_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if len(paths) >= limit:
                break

            key = row.get("File", "").strip()
            url = row.get("TIFF", "").strip()
            if not key or not url:
                continue

            suffix = Path(url).suffix or ".TIF"
            path = download_dir / f"{key}{suffix.lower()}"
            if force_download or not path.exists():
                print(f"Downloading {key}: {url}")
                with urllib.request.urlopen(url) as response, path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            paths.append(path)

    return paths


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def load_reference_png(tiff_path: Path, output_dir: Path) -> Image.Image:
    """Convert one TIFF image to an RGB square PNG reference."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{tiff_path.stem}_reference.png"
    with Image.open(tiff_path) as image:
        reference = center_crop_square(image.convert("RGB"))
    reference.save(png_path)
    return reference


def resize_square(image: Image.Image, size: int, method: str) -> Image.Image:
    return image.resize((size, size), RESAMPLE_FILTERS[method])


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_peak_groups(value: str | None) -> dict[int, tuple[int, ...]]:
    """Parse ``target:peak|peak,target:peak|peak`` or use defaults."""

    if value is None:
        return DEFAULT_PEAKS

    groups: dict[int, tuple[int, ...]] = {}
    for group in value.split(","):
        target_text, peaks_text = group.split(":", maxsplit=1)
        target_size = int(target_text.strip())
        peaks = tuple(int(item.strip()) for item in peaks_text.split("|") if item.strip())
        if not peaks:
            raise ValueError(f"target {target_size} has no peaks")
        groups[target_size] = peaks
    return groups


def source_sizes_for_peak(
    target_size: int,
    peak: int,
    *,
    k_values: list[int],
    min_source_size: int,
    sizes_per_peak: int,
) -> list[SynthesisCase]:
    """Build source sizes from ``k*C+d`` and ``k*C+(C-d)`` residues."""

    if peak <= 0 or peak >= target_size:
        raise ValueError(f"peak {peak} must be in (0, target_size={target_size})")

    rows: list[SynthesisCase] = []
    seen: set[int] = set()
    residues = ((peak, "d"), (target_size - peak, "C-d"))
    for k in k_values:
        for residue, family in residues:
            source_size = k * target_size + residue
            if source_size < min_source_size or source_size in seen:
                continue
            seen.add(source_size)
            rows.append(
                SynthesisCase(
                    target_size=target_size,
                    designed_peak=peak,
                    source_size=source_size,
                    residue=residue,
                    residue_family=family,
                )
            )

    rows.sort(key=lambda item: item.source_size)
    return rows[:sizes_per_peak]


def build_design(
    peak_groups: dict[int, tuple[int, ...]],
    *,
    k_values: list[int],
    min_source_size: int,
    sizes_per_peak: int,
) -> list[SynthesisCase]:
    design: list[SynthesisCase] = []
    for target_size in sorted(peak_groups):
        for peak in peak_groups[target_size]:
            design.extend(
                source_sizes_for_peak(
                    target_size,
                    peak,
                    k_values=k_values,
                    min_source_size=min_source_size,
                    sizes_per_peak=sizes_per_peak,
                )
            )
    return design


def synthesize_dataset(args: argparse.Namespace) -> Path:
    methods = [args.method]
    unknown_methods = sorted(set(methods) - set(RESAMPLE_FILTERS))
    if unknown_methods:
        raise ValueError(f"unknown interpolation methods: {', '.join(unknown_methods)}")

    if args.download:
        tiff_paths = download_tiff_paths(
            args.raise_csv,
            args.download_dir,
            args.limit_images,
            force_download=args.force_download,
        )
    else:
        tiff_paths = discover_tiff_paths(args.image_dir, args.raise_csv, args.limit_images)
    if not tiff_paths:
        raise RuntimeError("no TIFF images found or downloaded")

    peak_groups = parse_peak_groups(args.peak_groups)
    design = build_design(
        peak_groups,
        k_values=parse_int_list(args.k_values),
        min_source_size=args.min_source_size,
        sizes_per_peak=args.sizes_per_peak,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.outdir / "metadata.csv"

    with metadata_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_id",
                "source_size",
                "target_size",
                "method",
                "designed_peak",
                "residue",
                "residue_family",
                "source_path",
                "target_path",
            ]
        )

        for tiff_path in tiff_paths:
            reference = load_reference_png(tiff_path, args.outdir / "references")
            source_cache: dict[int, Path] = {}
            source_dir = args.outdir / "controlled_sources" / image_id(tiff_path)
            source_dir.mkdir(parents=True, exist_ok=True)

            for case in design:
                source_path = source_cache.get(case.source_size)
                if source_path is None:
                    source = resize_square(reference, case.source_size, args.source_method)
                    source_path = source_dir / f"source_{case.source_size}.png"
                    source.save(source_path)
                    source_cache[case.source_size] = source_path
                else:
                    source = Image.open(source_path).convert("RGB")

                target = resize_square(source, case.target_size, args.method)
                case_dir = (
                    args.outdir
                    / "cases"
                    / image_id(tiff_path)
                    / f"source_{case.source_size}_target_{case.target_size}_{args.method}"
                )
                case_dir.mkdir(parents=True, exist_ok=True)
                target_path = case_dir / "target.png"
                target.save(target_path)

                writer.writerow(
                    [
                        image_id(tiff_path),
                        case.source_size,
                        case.target_size,
                        args.method,
                        case.designed_peak,
                        case.residue,
                        case.residue_family,
                        source_path,
                        target_path,
                    ]
                )

    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("limit_images", type=int, nargs="?", default=100)
    parser.add_argument("--image-dir", type=Path, default=Path("test_images"))
    parser.add_argument("--raise-csv", type=Path, default=Path("RAISE_1k.csv"))
    parser.add_argument("--download", action="store_true", help="download TIFF files from the CSV TIFF column")
    parser.add_argument("--download-dir", type=Path, default=Path("test_results/raise_tiff_downloads"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("test_results/controlled_resampling_dataset"))
    parser.add_argument("--method", default="bicubic")
    parser.add_argument("--source-method", default="lanczos")
    parser.add_argument("--k-values", default="0,1,2")
    parser.add_argument("--sizes-per-peak", type=int, default=5)
    parser.add_argument("--min-source-size", type=int, default=128)
    parser.add_argument(
        "--peak-groups",
        default=None,
        help="optional format: '256:64|85|96,384:96|128|160,512:128|171|192'",
    )
    args = parser.parse_args()

    metadata_path = synthesize_dataset(args)
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
