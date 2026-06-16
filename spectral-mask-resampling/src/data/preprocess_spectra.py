import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

from src.data.make_patches import random_crop_rgb
from src.processing.residuals import tv_residual
from src.processing.spectrum import compute_log_rfft_spectrum
from src.processing.transforms import apply_jpeg_pil, resize_pil, rgb_to_y_float


CLASS_NAMES = [
    "original",
    "JPEG_Q80",
    "downsample_x8",
    "downsample_x16",
]
OBSERVED_SIZES = [128, 96, 64, 48, 32]


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


def process_patch(patch: Image.Image, class_index: int, args: argparse.Namespace) -> np.ndarray:
    if class_index == 0:
        img = patch
    elif class_index == 1:
        img = apply_jpeg_pil(patch, args.jpeg_quality)
    else:
        img = resize_pil(patch, args.observed_size, args.interpolation)
    y = rgb_to_y_float(img)
    residual = tv_residual(y, weight=args.tv_weight, max_num_iter=args.tv_max_iter)
    return compute_log_rfft_spectrum(
        residual,
        target_height=args.target_spectrum_height,
        target_width_rfft=args.target_spectrum_width_rfft,
        dc_sigma_bins=args.dc_sigma_bins,
    )


def selected_images(split_data: dict, split: str, limit_images: int | None) -> list[str]:
    images = split_data[split]
    if limit_images is not None:
        images = images[:limit_images]
    return images


def write_split(
    split: str,
    images: list[str],
    input_dir: Path | None,
    output_dir: Path,
    download_dir: Path,
    args: argparse.Namespace,
) -> dict:
    num_classes = len(CLASS_NAMES)
    samples = num_classes * len(args.observed_sizes) * args.samples_per_class_per_size
    shape = (samples, 1, args.target_spectrum_height, args.target_spectrum_width_rfft)
    spectra = open_memmap(output_dir / f"{split}_spectra.npy", mode="w+", dtype=args.dtype, shape=shape)
    labels = open_memmap(output_dir / f"{split}_labels.npy", mode="w+", dtype=np.int64, shape=(samples,))
    observed_sizes = open_memmap(output_dir / f"{split}_observed_sizes.npy", mode="w+", dtype=np.int64, shape=(samples,))

    rng = random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[split])
    index = 0
    source_cache = {}
    pbar = tqdm(total=samples, desc=f"preprocess {split}")
    for observed_size in args.observed_sizes:
        args.observed_size = observed_size
        for class_index, class_name in enumerate(CLASS_NAMES):
            factor = 1
            if class_name == "downsample_x8":
                factor = 8
            elif class_name == "downsample_x16":
                factor = 16
            source_patch_size = observed_size * factor
            made = 0
            while made < args.samples_per_class_per_size:
                source = rng.choice(images)
                if source not in source_cache:
                    source_cache[source] = resolve_image_path(source, input_dir, download_dir)
                with Image.open(source_cache[source]) as img:
                    if img.width < source_patch_size or img.height < source_patch_size:
                        continue
                    patch = random_crop_rgb(img, source_patch_size, rng)
                    spectra[index] = process_patch(patch, class_index, args).astype(args.dtype)
                    labels[index] = class_index
                    observed_sizes[index] = observed_size
                    index += 1
                    made += 1
                    pbar.update(1)
    pbar.close()

    spectra.flush()
    labels.flush()
    observed_sizes.flush()
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
    parser.add_argument("--dc-sigma-bins", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()

    if args.residual != "tv":
        raise ValueError(f"Unsupported residual: {args.residual}")
    if args.classes != CLASS_NAMES:
        raise ValueError(f"Version 1 expects classes: {CLASS_NAMES}")
    if args.observed_sizes != OBSERVED_SIZES:
        raise ValueError(f"Version 1 expects observed sizes: {OBSERVED_SIZES}")
    if args.jpeg_quality != 80:
        raise ValueError("Version 1 expects JPEG quality 80")
    if args.downsample_factors != [8, 16]:
        raise ValueError("Version 1 expects downsample factors 8 16")
    if args.target_spectrum_height != 512 or args.target_spectrum_width_rfft != 257:
        raise ValueError("Version 1 expects spectra with shape [1, 512, 257]")

    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.split_json).open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    metadata = {
        "class_names": CLASS_NAMES,
        "observed_sizes": args.observed_sizes,
        "samples_per_class_per_size": args.samples_per_class_per_size,
        "jpeg_quality": args.jpeg_quality,
        "downsample_factors": args.downsample_factors,
        "interpolation": args.interpolation,
        "residual": args.residual,
        "tv_weight": args.tv_weight,
        "tv_max_iter": args.tv_max_iter,
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
