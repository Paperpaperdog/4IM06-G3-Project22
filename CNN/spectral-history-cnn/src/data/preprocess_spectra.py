from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from src.processing.residuals import tv_residual
from src.processing.spectrum import compute_log_rfft_spectrum
from src.processing.transforms import apply_jpeg_pil, random_crop, resize_to_final, rgb_to_y_float
from src.utils.io import ensure_dir, load_json, load_yaml, save_json, update_nested

ImageFile.LOAD_TRUNCATED_IMAGES = True


CLASS_CROP_SIZES = {
    "original": 64,
    "JPEG": 64,
    "downsample_x2": 128,
    "downsample_x4": 256,
    "downsample_x8": 512,
    "downsample_x16": 1024,
}


def resolve_image_path(raise_dir: Path, rel_path: str) -> Path:
    candidate = raise_dir / rel_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image listed in split JSON is missing: {candidate}")


def cache_filename_from_url(url: str, file_id: str | None = None) -> str:
    suffix = Path(urlparse(url).path).suffix or ".img"
    if file_id:
        return f"{file_id}{suffix}"
    name = Path(urlparse(url).path).name
    return name if name else f"downloaded{suffix}"


def download_url_to_cache(url: str, cache_dir: Path, file_id: str | None = None, timeout: int = 120) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / cache_filename_from_url(url, file_id)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(target_path)
    return target_path


def resolve_source_entry(
    source_entry: str | Dict[str, Any],
    raise_dir: Path | None,
    image_cache_dir: Path,
) -> tuple[Path, Dict[str, Any]]:
    if isinstance(source_entry, str):
        if raise_dir is None:
            raise ValueError("Split JSON contains local relative paths but no raise_dir is configured.")
        source_path = resolve_image_path(raise_dir, source_entry)
        return source_path, {
            "source_file_id": source_path.stem,
            "source_url": "",
            "source_csv_index": "",
            "download_cache_path": "",
        }

    url = str(source_entry.get("url", "")).strip()
    file_id = str(source_entry.get("file_id", "")).strip() or None
    if not url:
        raise ValueError(f"URL split entry is missing 'url': {source_entry}")

    source_path = download_url_to_cache(url, image_cache_dir, file_id=file_id)
    return source_path, {
        "source_file_id": file_id or source_path.stem,
        "source_url": url,
        "source_csv_index": source_entry.get("csv_index", ""),
        "download_cache_path": str(source_path),
    }


def make_final_image(
    source_path: Path,
    source_metadata: Dict[str, Any],
    class_name: str,
    final_size: int,
    jpeg_quality: int,
    interpolation: str,
    rng: np.random.Generator,
) -> tuple[Image.Image, Dict[str, Any]] | None:
    crop_size = CLASS_CROP_SIZES[class_name]
    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            cropped = random_crop(img, crop_size, rng)
    except Exception:
        return None

    if cropped is None:
        return None

    crop, crop_x, crop_y = cropped
    jpeg_value = ""
    interpolation_value = ""

    if class_name == "JPEG":
        final_img = apply_jpeg_pil(crop, jpeg_quality)
        jpeg_value = int(jpeg_quality)
    elif class_name.startswith("downsample"):
        final_img = resize_to_final(crop, final_size, interpolation)
        interpolation_value = interpolation
    else:
        final_img = crop

    if final_img.size != (final_size, final_size):
        final_img = resize_to_final(final_img, final_size, interpolation)

    metadata = {
        "source_filename": source_path.name,
        "source_path": str(source_path),
        "crop_size": crop_size,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "jpeg_quality": jpeg_value,
        "interpolation": interpolation_value,
    }
    metadata.update(source_metadata)
    return final_img.convert("RGB"), metadata


def sample_one_spectrum(
    source_entry: str | Dict[str, Any],
    class_id: int,
    class_name: str,
    split: str,
    config: Dict[str, Any],
    raise_dir: Path | None,
    image_cache_dir: Path,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Dict[str, Any]] | None:
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]

    try:
        source_path, source_metadata = resolve_source_entry(source_entry, raise_dir, image_cache_dir)
    except Exception as exc:
        print(f"Warning: failed to resolve source {source_entry}: {exc}")
        return None

    result = make_final_image(
        source_path=source_path,
        source_metadata=source_metadata,
        class_name=class_name,
        final_size=int(data_cfg["final_size"]),
        jpeg_quality=int(data_cfg["jpeg_quality"]),
        interpolation=str(data_cfg["interpolation"]),
        rng=rng,
    )
    if result is None:
        return None
    final_img, metadata = result

    y = rgb_to_y_float(final_img)
    residual = tv_residual(
        y,
        weight=float(prep_cfg["tv_weight"]),
        eps=float(prep_cfg["tv_eps"]),
        max_num_iter=int(prep_cfg["tv_max_num_iter"]),
    )
    spectrum = compute_log_rfft_spectrum(residual, dc_sigma_bins=float(prep_cfg["dc_sigma_bins"]))

    metadata.update(
        {
            "split": split,
            "class_id": class_id,
            "class_name": class_name,
            "random_seed": int(config["data"]["seed"]),
        }
    )
    return spectrum, metadata


def build_split(
    split: str,
    sources: list[str | Dict[str, Any]],
    config: Dict[str, Any],
    processed_dir: Path,
    raise_dir: Path | None,
    image_cache_dir: Path,
    limit_samples: int | None,
) -> None:
    class_names = list(config["data"]["classes"])
    final_size = int(config["data"]["final_size"])
    width_rfft = final_size // 2 + 1
    target_key = f"{split}_samples_per_class"
    target_per_class = int(config["data"][target_key])
    if limit_samples is not None:
        target_per_class = min(target_per_class, int(limit_samples))

    total_samples = target_per_class * len(class_names)
    spectra = np.empty((total_samples, 1, final_size, width_rfft), dtype=np.float16)
    labels = np.empty((total_samples,), dtype=np.int64)
    metadata_rows: list[Dict[str, Any]] = []

    split_seed_offsets = {"train": 0, "val": 100_000, "test": 200_000}
    base_seed = int(config["data"]["seed"]) + split_seed_offsets[split]
    out_idx = 0

    for class_id, class_name in enumerate(class_names):
        if class_name not in CLASS_CROP_SIZES:
            raise ValueError(f"Unknown class '{class_name}'.")

        rng = np.random.default_rng(base_seed + class_id * 1009)
        progress = tqdm(total=target_per_class, desc=f"{split}:{class_name}", unit="sample")
        made = 0
        attempts = 0
        max_attempts = max(target_per_class * 500, 5000)

        while made < target_per_class:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could only generate {made}/{target_per_class} samples for {split}:{class_name}. "
                    f"Many images may be smaller than crop size {CLASS_CROP_SIZES[class_name]}."
                )
            attempts += 1
            source_entry = sources[int(rng.integers(0, len(sources)))]
            sample = sample_one_spectrum(
                source_entry,
                class_id,
                class_name,
                split,
                config,
                raise_dir,
                image_cache_dir,
                rng,
            )
            if sample is None:
                continue

            spectrum, metadata = sample
            spectra[out_idx] = spectrum.astype(np.float16)
            labels[out_idx] = class_id
            metadata_rows.append(metadata)
            out_idx += 1
            made += 1
            progress.update(1)
        progress.close()

    np.save(processed_dir / f"{split}_spectra.npy", spectra)
    np.save(processed_dir / f"{split}_labels.npy", labels)
    pd.DataFrame(metadata_rows).to_csv(processed_dir / f"{split}_metadata.csv", index=False)
    print(f"Saved {split}: spectra={spectra.shape}, labels={labels.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final64 TV-rFFT spectrum cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--raise-dir", default=None)
    parser.add_argument("--image-cache-dir", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--limit-samples", type=int, default=None, help="Cap samples per class for debug runs.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    update_nested(config, "paths", "raise_dir", args.raise_dir)
    update_nested(config, "paths", "image_cache_dir", args.image_cache_dir)
    update_nested(config, "paths", "split_json", args.split_json)
    update_nested(config, "paths", "processed_dir", args.processed_dir)

    split_json_path = Path(config["paths"]["split_json"])
    split_data = load_json(split_json_path)

    raise_dir = None
    configured_raise_dir = config["paths"].get("raise_dir")
    if configured_raise_dir:
        candidate = Path(configured_raise_dir).expanduser()
        if candidate.exists():
            raise_dir = candidate
    if raise_dir is None and split_data.get("input_dir"):
        candidate = Path(split_data["input_dir"]).expanduser()
        if candidate.exists():
            raise_dir = candidate

    image_cache_dir = ensure_dir(config["paths"].get("image_cache_dir", "data/raw/raise_tiff"))

    processed_dir = ensure_dir(config["paths"]["processed_dir"])
    save_json(config, processed_dir / "preprocess_config.json")
    with (processed_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(config["data"]["classes"], f, indent=2)

    for split in ("train", "val", "test"):
        sources = split_data["splits"][split]
        build_split(split, sources, config, processed_dir, raise_dir, image_cache_dir, args.limit_samples)


if __name__ == "__main__":
    main()
