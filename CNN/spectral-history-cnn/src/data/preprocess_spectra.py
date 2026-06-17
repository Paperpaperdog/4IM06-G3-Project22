from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# Minimum crop size (pixels) allowed for an upsampling class, so that an
# upsample factor on a small final_size does not crop a degenerate patch.
MIN_UPSAMPLE_CROP = 4

SPLIT_SEED_OFFSETS = {"train": 0, "val": 100_000, "test": 200_000}


def parse_class_spec(class_name: str) -> tuple[str, int]:
    """Map a class name to ``(kind, factor)``.

    Supported kinds: ``original`` / ``jpeg`` (factor 1), ``downsample`` and
    ``upsample`` (factor parsed from the ``_xN`` suffix). This replaces the old
    hard-coded crop table so that input size and up/down direction are derived
    from the class name and the configured ``final_size``.
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
        f"Unknown class '{class_name}'. Expected 'original', 'JPEG', "
        "'downsample_xN' or 'upsample_xN'."
    )


def class_crop_size(class_name: str, final_size: int) -> int:
    """Source crop size for a class given the final observed size.

    - original / JPEG: crop exactly ``final_size``.
    - downsample_xN: crop ``N * final_size`` then resize down to ``final_size``.
    - upsample_xN: crop ``final_size // N`` then resize up to ``final_size``.
    """
    kind, factor = parse_class_spec(class_name)
    if kind in ("original", "jpeg"):
        return int(final_size)
    if kind == "downsample":
        return int(final_size) * int(factor)
    crop = int(final_size) // int(factor)
    if crop < MIN_UPSAMPLE_CROP:
        raise ValueError(
            f"Upsample class '{class_name}' with final_size={final_size} yields a "
            f"degenerate crop of {crop}px (< {MIN_UPSAMPLE_CROP}). "
            "Use a smaller factor or larger final_size."
        )
    return crop


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
    crop_size = class_crop_size(class_name, final_size)
    kind, _factor = parse_class_spec(class_name)
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

    if kind == "jpeg":
        final_img = apply_jpeg_pil(crop, jpeg_quality)
        jpeg_value = int(jpeg_quality)
    elif kind in ("downsample", "upsample"):
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


def build_class_samples(
    split: str,
    class_id: int,
    class_name: str,
    sources: list[str | Dict[str, Any]],
    config: Dict[str, Any],
    raise_dir: Path | None,
    image_cache_dir: Path,
    limit_samples: int | None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    data_cfg = config["data"]
    final_size = int(data_cfg["final_size"])
    parse_class_spec(class_name)  # validate, raises on unknown class
    width_rfft = final_size // 2 + 1
    target_key = f"{split}_samples_per_class"
    target_per_class = int(data_cfg[target_key])
    if limit_samples is not None:
        target_per_class = min(target_per_class, int(limit_samples))

    spectra = np.empty((target_per_class, 1, final_size, width_rfft), dtype=np.float16)
    labels = np.full((target_per_class,), class_id, dtype=np.int64)
    metadata_rows: list[Dict[str, Any]] = []

    base_seed = int(config["data"]["seed"]) + SPLIT_SEED_OFFSETS[split]
    rng = np.random.default_rng(base_seed + class_id * 1009)
    progress = tqdm(
        total=target_per_class,
        desc=f"{split}:{class_name}",
        unit="sample",
        disable=not show_progress,
    )
    made = 0
    attempts = 0
    max_attempts = max(target_per_class * 500, 5000)

    while made < target_per_class:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could only generate {made}/{target_per_class} samples for {split}:{class_name}. "
                f"Many images may be smaller than crop size {class_crop_size(class_name, final_size)}."
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
        spectra[made] = spectrum.astype(np.float16)
        metadata_rows.append(metadata)
        made += 1
        progress.update(1)
    progress.close()
    return spectra, labels, metadata_rows


def _build_class_samples_worker(
    task: Dict[str, Any],
) -> tuple[str, int, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    raise_dir = Path(task["raise_dir"]) if task["raise_dir"] else None
    image_cache_dir = Path(task["image_cache_dir"])
    spectra, labels, metadata_rows = build_class_samples(
        split=task["split"],
        class_id=int(task["class_id"]),
        class_name=str(task["class_name"]),
        sources=task["sources"],
        config=task["config"],
        raise_dir=raise_dir,
        image_cache_dir=image_cache_dir,
        limit_samples=task["limit_samples"],
        show_progress=False,
    )
    return str(task["split"]), int(task["class_id"]), spectra, labels, metadata_rows


def build_split(
    split: str,
    sources: list[str | Dict[str, Any]],
    config: Dict[str, Any],
    processed_dir: Path,
    raise_dir: Path | None,
    image_cache_dir: Path,
    limit_samples: int | None,
    workers: int = 1,
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

    if workers <= 1:
        out_idx = 0
        for class_id, class_name in enumerate(class_names):
            class_spectra, class_labels, class_metadata = build_class_samples(
                split,
                class_id,
                class_name,
                sources,
                config,
                raise_dir,
                image_cache_dir,
                limit_samples,
                show_progress=True,
            )
            end_idx = out_idx + len(class_labels)
            spectra[out_idx:end_idx] = class_spectra
            labels[out_idx:end_idx] = class_labels
            metadata_rows.extend(class_metadata)
            out_idx = end_idx
    else:
        tasks = [
            {
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "sources": sources,
                "config": config,
                "raise_dir": str(raise_dir) if raise_dir is not None else None,
                "image_cache_dir": str(image_cache_dir),
                "limit_samples": limit_samples,
            }
            for class_id, class_name in enumerate(class_names)
        ]
        class_results: dict[int, tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]] = {}
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_build_class_samples_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split}:classes", unit="class"):
                _, class_id, class_spectra, class_labels, class_metadata = future.result()
                class_results[class_id] = (class_spectra, class_labels, class_metadata)

        out_idx = 0
        for class_id in range(len(class_names)):
            class_spectra, class_labels, class_metadata = class_results[class_id]
            end_idx = out_idx + len(class_labels)
            spectra[out_idx:end_idx] = class_spectra
            labels[out_idx:end_idx] = class_labels
            metadata_rows.extend(class_metadata)
            out_idx = end_idx

    np.save(processed_dir / f"{split}_spectra.npy", spectra)
    np.save(processed_dir / f"{split}_labels.npy", labels)
    pd.DataFrame(metadata_rows).to_csv(processed_dir / f"{split}_metadata.csv", index=False)
    print(f"Saved {split}: spectra={spectra.shape}, labels={labels.shape}")


def build_all_splits_parallel(
    split_data: Dict[str, Any],
    config: Dict[str, Any],
    processed_dir: Path,
    raise_dir: Path | None,
    image_cache_dir: Path,
    limit_samples: int | None,
    workers: int,
) -> None:
    class_names = list(config["data"]["classes"])
    tasks = []
    for split in ("train", "val", "test"):
        sources = split_data["splits"][split]
        for class_id, class_name in enumerate(class_names):
            tasks.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "sources": sources,
                    "config": config,
                    "raise_dir": str(raise_dir) if raise_dir is not None else None,
                    "image_cache_dir": str(image_cache_dir),
                    "limit_samples": limit_samples,
                }
            )

    split_class_results: dict[str, dict[int, tuple[np.ndarray, np.ndarray, list[Dict[str, Any]]]]] = {
        split: {} for split in ("train", "val", "test")
    }
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        futures = [executor.submit(_build_class_samples_worker, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="preprocess", unit="task"):
            split_name, class_id, class_spectra, class_labels, class_metadata = future.result()
            split_class_results[split_name][class_id] = (class_spectra, class_labels, class_metadata)

    for split in ("train", "val", "test"):
        class_results = split_class_results[split]
        chunks = [class_results[class_id] for class_id in range(len(class_names))]
        spectra = np.concatenate([item[0] for item in chunks], axis=0)
        labels = np.concatenate([item[1] for item in chunks], axis=0)
        metadata_rows: list[Dict[str, Any]] = []
        for _, _, class_metadata in chunks:
            metadata_rows.extend(class_metadata)

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process count for parallel class generation within each split (e.g. 6 or 32).",
    )
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

    workers = max(1, int(args.workers))
    if workers > 1:
        print(f"Parallel preprocess with workers={workers} (train/val/test x classes)")
        build_all_splits_parallel(
            split_data,
            config,
            processed_dir,
            raise_dir,
            image_cache_dir,
            args.limit_samples,
            workers,
        )
    else:
        for split in ("train", "val", "test"):
            sources = split_data["splits"][split]
            build_split(
                split,
                sources,
                config,
                processed_dir,
                raise_dir,
                image_cache_dir,
                args.limit_samples,
                workers=1,
            )


if __name__ == "__main__":
    main()
