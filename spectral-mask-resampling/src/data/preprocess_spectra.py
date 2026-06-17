import argparse
import json
import os
import random
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

from src.data.make_patches import random_crop_rgb
from src.processing.residuals import tv_residual
from src.processing.spectrum import compute_log_rfft_spectrum
from src.processing.transforms import apply_jpeg_pil, resize_pil, rgb_to_y_float


# Unified 6-class set shared with the CNN pipeline (adds upsampling so the two
# learnable methods are directly comparable). The legacy 4-class set is a subset.
CLASS_NAMES = [
    "original",
    "JPEG_Q80",
    "downsample_x8",
    "downsample_x16",
    "upsample_x2",
    "upsample_x4",
]
OBSERVED_SIZES = [128, 96, 64, 48, 32]

# Minimum source crop (pixels) for an upsampling class to avoid degenerate crops.
MIN_UPSAMPLE_CROP = 4


def parse_class_spec(class_name: str) -> tuple[str, int]:
    """Map a class name to ``(kind, factor)``.

    kinds: ``original`` / ``jpeg`` (factor 1), ``downsample`` (crop o*factor),
    ``upsample`` (crop o//factor). Both then resample to the observed size o.
    """
    if class_name == "original":
        return "original", 1
    if class_name in ("JPEG", "JPEG_Q80"):
        return "jpeg", 1
    if class_name.startswith("downsample_x"):
        return "downsample", int(class_name.rsplit("_x", 1)[-1])
    if class_name.startswith("upsample_x"):
        return "upsample", int(class_name.rsplit("_x", 1)[-1])
    raise ValueError(
        f"Unknown class '{class_name}'. Expected 'original', 'JPEG_Q80', "
        "'downsample_xN' or 'upsample_xN'."
    )


def source_patch_size(class_name: str, observed_size: int) -> int:
    """Source crop size for a class at a given observed size."""
    kind, factor = parse_class_spec(class_name)
    if kind in ("original", "jpeg"):
        return int(observed_size)
    if kind == "downsample":
        return int(observed_size) * int(factor)
    crop = int(observed_size) // int(factor)
    if crop < MIN_UPSAMPLE_CROP:
        raise ValueError(
            f"Upsample class '{class_name}' with observed_size={observed_size} yields a "
            f"degenerate crop of {crop}px (< {MIN_UPSAMPLE_CROP})."
        )
    return crop


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


def process_patch(patch: Image.Image, class_name: str, args: argparse.Namespace) -> np.ndarray:
    kind, _factor = parse_class_spec(class_name)
    if kind == "original":
        img = patch
    elif kind == "jpeg":
        img = apply_jpeg_pil(patch, args.jpeg_quality)
    else:
        # downsample (source patch > o) and upsample (source patch < o) both
        # resample the cropped patch to the observed size o.
        img = resize_pil(patch, args.observed_size, args.interpolation)
    y = rgb_to_y_float(img)
    residual = tv_residual(y, weight=args.tv_weight, max_num_iter=args.tv_max_iter)
    # Native mode keeps the per-size rFFT resolution (no 512x257 remap), so the
    # observed image is o x o and the spectrum is (o, o//2+1).
    if getattr(args, "native_spectrum", False):
        target_height = None
        target_width_rfft = None
    else:
        target_height = args.target_spectrum_height
        target_width_rfft = args.target_spectrum_width_rfft
    return compute_log_rfft_spectrum(
        residual,
        target_height=target_height,
        target_width_rfft=target_width_rfft,
        dc_sigma_bins=args.dc_sigma_bins,
    )


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
    args.observed_size = task["observed_size"]
    class_name = task["class_name"]
    class_index = task["class_index"]
    crop_size = task["crop_size"]
    count = task["count"]
    start = task["start"]
    images = task["images"]
    input_dir = task["input_dir"]
    download_dir = task["download_dir"]

    rng = random.Random(task["block_seed"])
    spectra = open_memmap(task["spectra_path"], mode="r+")
    labels = open_memmap(task["labels_path"], mode="r+")
    observed_sizes = open_memmap(task["sizes_path"], mode="r+")

    source_cache: dict[str, Path] = {}
    made = 0
    while made < count:
        source = rng.choice(images)
        if source not in source_cache:
            source_cache[source] = resolve_image_path(source, input_dir, download_dir)
        with Image.open(source_cache[source]) as img:
            if img.width < crop_size or img.height < crop_size:
                continue
            patch = random_crop_rgb(img, crop_size, rng)
            spectra[start + made] = process_patch(patch, class_name, args).astype(args.dtype)
            labels[start + made] = class_index
            observed_sizes[start + made] = task["observed_size"]
            made += 1

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
    shape = (samples, 1, args.target_spectrum_height, args.target_spectrum_width_rfft)

    spectra_path = output_dir / f"{split}_spectra.npy"
    labels_path = output_dir / f"{split}_labels.npy"
    sizes_path = output_dir / f"{split}_observed_sizes.npy"

    # Pre-create the memmaps in the parent, then close them so workers can each
    # open the same files and write their own disjoint row ranges.
    spectra = open_memmap(spectra_path, mode="w+", dtype=args.dtype, shape=shape)
    labels = open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(samples,))
    observed_sizes = open_memmap(sizes_path, mode="w+", dtype=np.int64, shape=(samples,))
    del spectra, labels, observed_sizes

    split_seed = args.seed + {"train": 0, "val": 1, "test": 2}[split]

    tasks: list[dict] = []
    block_pos = 0
    for size_idx, observed_size in enumerate(args.observed_sizes):
        for class_index, class_name in enumerate(class_names):
            crop_size = source_patch_size(class_name, observed_size)
            tasks.append(
                {
                    "block_pos": block_pos,
                    "start": block_pos * spc,
                    "count": spc,
                    "observed_size": observed_size,
                    "class_name": class_name,
                    "class_index": class_index,
                    "crop_size": crop_size,
                    # Deterministic per-block seed (independent of worker count).
                    "block_seed": split_seed * 100003 + size_idx * 1009 + class_index,
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
    parser.add_argument("--target-spectrum-height", type=int, default=512)
    parser.add_argument("--target-spectrum-width-rfft", type=int, default=257)
    parser.add_argument(
        "--native-spectrum",
        action="store_true",
        help="Keep the native per-size rFFT resolution (o, o//2+1) instead of "
             "remapping to the 512x257 grid. Requires a single observed size.",
    )
    parser.add_argument("--dc-sigma-bins", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
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
    # downsample_xN / upsample_xN). Class set, observed sizes and factors are no
    # longer hard-coded, so per-size and upsampling experiments are allowed.
    for class_name in args.classes:
        parse_class_spec(class_name)
    if args.native_spectrum:
        # Native spectra differ in shape per observed size, so one cache (one
        # memmap) can only hold a single size. Per-size sweep configs satisfy this.
        if len(args.observed_sizes) != 1:
            raise ValueError(
                "--native-spectrum requires exactly one observed size per config "
                f"(got {args.observed_sizes}); use one per-size config each."
            )
        o = int(args.observed_sizes[0])
        args.target_spectrum_height = o
        args.target_spectrum_width_rfft = o // 2 + 1
    elif args.target_spectrum_height != 512 or args.target_spectrum_width_rfft != 257:
        raise ValueError("The normalized frequency grid must be [1, 512, 257]")

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
        "native_spectrum": bool(args.native_spectrum),
        "target_spectrum_height": args.target_spectrum_height,
        "target_spectrum_width_rfft": args.target_spectrum_width_rfft,
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
