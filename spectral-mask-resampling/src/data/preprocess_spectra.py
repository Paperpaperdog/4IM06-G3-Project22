import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.unified_protocol import (  # noqa: E402
    CANONICAL_CLASSES,
    DEFAULT_SEED,
    make_aligned_observed_patch,
    required_source_crop,
)
from src.processing.residuals import tv_residual
from src.processing.spectrum import compute_log_rfft_spectrum
from src.processing.transforms import rgb_to_y_float

CLASS_NAMES = list(CANONICAL_CLASSES)
OBSERVED_SIZES = [128, 96, 64, 32]


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def cached_url_path(url: str, download_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name
    return download_dir / filename


def resolve_image_path(source: str, input_dir: Path | None, download_dir: Path) -> Path:
    if is_url(source):
        download_dir.mkdir(parents=True, exist_ok=True)
        path = cached_url_path(source, download_dir)
        if not path.exists():
            urllib.request.urlretrieve(source, path)
        return path
    if input_dir is None:
        return Path(source)
    return input_dir / source


def process_observed_patch(patch: Image.Image, args: argparse.Namespace) -> np.ndarray:
    y = rgb_to_y_float(patch)
    residual = tv_residual(y, weight=args.tv_weight, max_num_iter=args.tv_max_iter)
    return compute_log_rfft_spectrum(residual, dc_sigma_bins=args.dc_sigma_bins)


def selected_images(split_data: dict, split: str, limit_images: int | None) -> list[str]:
    images = split_data[split]
    if limit_images is not None:
        images = images[:limit_images]
    return images


def _generate_block(task: dict) -> tuple[int, int]:
    """Generate one (observed_size, class) block and write its disjoint slice.

    Runs in a worker process. Opens the pre-created memmaps in ``r+`` mode and
    writes only its own contiguous ``[start:start+count)`` region, so workers
    never touch overlapping rows and no locking is required. Returns
    ``(block_pos, count)`` for progress accounting.
    """
    args = task["args"]
    observed_size = task["observed_size"]
    class_name = task["class_name"]
    class_index = task["class_index"]
    split = task["split"]
    count = task["count"]
    start = task["start"]
    images = task["images"]
    input_dir = task["input_dir"]
    download_dir = task["download_dir"]

    spectra = open_memmap(task["spectra_path"], mode="r+")
    labels = open_memmap(task["labels_path"], mode="r+")
    observed_sizes = open_memmap(task["sizes_path"], mode="r+")

    source_cache: dict[str, Path] = {}

    def open_source(entry: str) -> Image.Image | None:
        if entry not in source_cache:
            source_cache[entry] = resolve_image_path(entry, input_dir, download_dir)
        with Image.open(source_cache[entry]) as img:
            return img.convert("RGB").copy()

    for made in range(count):
        sample_index = made
        patch = make_aligned_observed_patch(
            images,
            open_source,
            class_name,
            observed_size,
            args.seed,
            split,
            class_index,
            sample_index,
            args.jpeg_quality,
        )
        if patch is None:
            raise RuntimeError(
                f"Failed to generate sample {sample_index} for {split}:{class_name} "
                f"at observed_size={observed_size}"
            )
        spectra[start + made] = process_observed_patch(patch, args).astype(args.dtype)
        labels[start + made] = class_index
        observed_sizes[start + made] = observed_size

    spectra.flush()
    labels.flush()
    observed_sizes.flush()
    return task["block_pos"], count


def write_split(
    split: str,
    images: list[str],
    input_dir: Path | None,
    output_dir: Path,
    download_dir: Path,
    args: argparse.Namespace,
) -> dict:
    class_names = args.classes
    num_classes = len(class_names)
    spc = args.samples_per_class_per_size
    samples = num_classes * len(args.observed_sizes) * spc
    # Native spectrum: one observed size per cache, shape (o, o//2+1).
    o = int(args.observed_sizes[0])
    shape = (samples, 1, o, o // 2 + 1)

    spectra_path = output_dir / f"{split}_spectra.npy"
    labels_path = output_dir / f"{split}_labels.npy"
    sizes_path = output_dir / f"{split}_observed_sizes.npy"

    # Pre-create the memmaps in the parent, then close them so workers can each
    # open the same files and write their own disjoint row ranges.
    spectra = open_memmap(spectra_path, mode="w+", dtype=args.dtype, shape=shape)
    labels = open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(samples,))
    observed_sizes = open_memmap(sizes_path, mode="w+", dtype=np.int64, shape=(samples,))
    del spectra, labels, observed_sizes

    tasks: list[dict] = []
    block_pos = 0
    for size_idx, observed_size in enumerate(args.observed_sizes):
        for class_index, class_name in enumerate(class_names):
            required_source_crop(class_name, observed_size)
            tasks.append(
                {
                    "block_pos": block_pos,
                    "start": block_pos * spc,
                    "count": spc,
                    "split": split,
                    "observed_size": observed_size,
                    "class_name": class_name,
                    "class_index": class_index,
                    "images": images,
                    "input_dir": input_dir,
                    "download_dir": download_dir,
                    "spectra_path": spectra_path,
                    "labels_path": labels_path,
                    "sizes_path": sizes_path,
                    "args": args,
                }
            )
            block_pos += 1

    workers = (os.cpu_count() or 1) if args.workers == 0 else max(1, args.workers)
    workers = min(workers, len(tasks))

    pbar = tqdm(total=samples, desc=f"preprocess {split}")
    if workers == 1:
        for task in tasks:
            _, count = _generate_block(task)
            pbar.update(count)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_generate_block, task) for task in tasks]
            for future in as_completed(futures):
                _, count = future.result()
                pbar.update(count)
    pbar.close()

    return {"split": split, "images": len(images), "samples": samples, "shape": list(shape)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download-dir", default="data/raw/raise_tiff")
    parser.add_argument("--classes", nargs="+", default=CLASS_NAMES)
    parser.add_argument("--observed-sizes", type=int, nargs="+", default=OBSERVED_SIZES)
    parser.add_argument("--samples-per-class-per-size", type=int, default=1000)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--downsample-factors", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--interpolation", default="bicubic")
    parser.add_argument("--residual", default="tv")
    parser.add_argument("--tv-weight", type=float, default=0.08)
    parser.add_argument("--tv-max-iter", type=int, default=30)
    parser.add_argument("--dc-sigma-bins", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit-images", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes over (size, class) blocks. 0 = all CPU cores, 1 = single process.",
    )
    args = parser.parse_args()

    if args.residual != "tv":
        raise ValueError(f"Unsupported residual: {args.residual}")
    # Validate that every requested class is supported (original / JPEG /
    # downsample_xN / upsample_xN).
    for class_name in args.classes:
        required_source_crop(class_name, args.observed_sizes[0])
    # Native spectra differ in shape per observed size, so one cache (one memmap)
    # holds a single size. Per-size sweep configs satisfy this by construction.
    if len(args.observed_sizes) != 1:
        raise ValueError(
            "Native-spectrum preprocessing requires exactly one observed size per "
            f"config (got {args.observed_sizes}); use one per-size config each."
        )

    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.split_json).open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    metadata = {
        "class_names": args.classes,
        "observed_sizes": args.observed_sizes,
        "samples_per_class_per_size": args.samples_per_class_per_size,
        "jpeg_quality": args.jpeg_quality,
        "downsample_factors": args.downsample_factors,
        "interpolation": args.interpolation,
        "residual": args.residual,
        "tv_weight": args.tv_weight,
        "tv_max_iter": args.tv_max_iter,
        "native_spectrum": True,
        "spectrum_height": int(args.observed_sizes[0]),
        "spectrum_width_rfft": int(args.observed_sizes[0]) // 2 + 1,
        "dc_sigma_bins": args.dc_sigma_bins,
        "seed": args.seed,
        "dtype": args.dtype,
        "download_dir": str(download_dir),
        "source": split_data.get("source", {}),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        images = selected_images(split_data, split, args.limit_images)
        metadata["splits"][split] = write_split(split, images, input_dir, output_dir, download_dir, args)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
