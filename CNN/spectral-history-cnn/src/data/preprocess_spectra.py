from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.unified_protocol import (  # noqa: E402
    make_aligned_observed_patch,
    normalize_split_data,
    required_source_crop,
)
from src.processing.residuals import tv_residual
from src.processing.spectrum import compute_log_rfft_spectrum
from src.processing.transforms import rgb_to_y_float
from src.utils.config_guard import reject_legacy_config_path
from src.utils.io import ensure_dir, load_json, load_yaml, save_json, update_nested

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def sample_one_spectrum(
    sources: list[str | Dict[str, Any]],
    sample_index: int,
    class_id: int,
    class_name: str,
    split: str,
    config: Dict[str, Any],
    raise_dir: Path | None,
    image_cache_dir: Path,
) -> tuple[np.ndarray, Dict[str, Any]] | None:
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    final_size = int(data_cfg["final_size"])
    global_seed = int(data_cfg["seed"])

    def open_source(entry: str | Dict[str, Any]) -> Image.Image | None:
        try:
            source_path, _source_metadata = resolve_source_entry(entry, raise_dir, image_cache_dir)
            with Image.open(source_path) as img:
                return img.convert("RGB").copy()
        except Exception:
            return None

    patch = make_aligned_observed_patch(
        sources,
        open_source,
        class_name,
        final_size,
        global_seed,
        split,
        class_id,
        sample_index,
        int(data_cfg["jpeg_quality"]),
    )
    if patch is None:
        return None

    y = rgb_to_y_float(patch)
    residual = tv_residual(
        y,
        weight=float(prep_cfg["tv_weight"]),
        eps=float(prep_cfg["tv_eps"]),
        max_num_iter=int(prep_cfg["tv_max_num_iter"]),
    )
    spectrum = compute_log_rfft_spectrum(residual, dc_sigma_bins=float(prep_cfg["dc_sigma_bins"]))
    metadata = {
        "split": split,
        "class_id": class_id,
        "class_name": class_name,
        "sample_index": sample_index,
        "random_seed": global_seed,
    }
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
    required_source_crop(class_name, final_size)
    width_rfft = final_size // 2 + 1
    target_key = f"{split}_samples_per_class"
    target_per_class = int(data_cfg[target_key])
    if limit_samples is not None:
        target_per_class = min(target_per_class, int(limit_samples))

    spectra = np.empty((target_per_class, 1, final_size, width_rfft), dtype=np.float16)
    labels = np.full((target_per_class,), class_id, dtype=np.int64)
    metadata_rows: list[Dict[str, Any]] = []

    progress = tqdm(
        total=target_per_class,
        desc=f"{split}:{class_name}",
        unit="sample",
        disable=not show_progress,
    )
    for made in range(target_per_class):
        sample = sample_one_spectrum(
            sources,
            made,
            class_id,
            class_name,
            split,
            config,
            raise_dir,
            image_cache_dir,
        )
        if sample is None:
            raise RuntimeError(
                f"Failed to generate sample {made} for {split}:{class_name} "
                f"at final_size={final_size}"
            )
        spectrum, metadata = sample
        spectra[made] = spectrum.astype(np.float16)
        metadata_rows.append(metadata)
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
    split_lists = normalize_split_data(split_data)
    tasks = []
    for split in ("train", "val", "test"):
        sources = split_lists[split]
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

    reject_legacy_config_path(args.config)
    config = load_yaml(args.config)
    update_nested(config, "paths", "raise_dir", args.raise_dir)
    update_nested(config, "paths", "image_cache_dir", args.image_cache_dir)
    update_nested(config, "paths", "split_json", args.split_json)
    update_nested(config, "paths", "processed_dir", args.processed_dir)

    split_json_path = Path(config["paths"]["split_json"])
    split_data = load_json(split_json_path)
    split_lists = normalize_split_data(split_data)

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
            sources = split_lists[split]
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
