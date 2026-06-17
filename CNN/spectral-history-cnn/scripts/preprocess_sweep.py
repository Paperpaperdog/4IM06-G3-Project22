#!/usr/bin/env python3
"""Generate a CNN spectrum cache for ONE observed size of the unified sweep.

Unlike the original ``src/data/preprocess_spectra.py`` (locked to 64x64, 6
downsample-only classes), this script:

* uses the shared :mod:`experiments.unified_protocol` (6 classes incl. upsampling),
* supports an arbitrary observed size ``o`` (native spectrum ``[1, o, o//2+1]``),
* reads the SAME split JSON used by every route so the comparison is fair.

Output (compatible with ``src.data.dataset.SpectraDataset``):

  <output_dir>/{train,val,test}_spectra.npy   float16 [N, 1, o, o//2+1]
  <output_dir>/{train,val,test}_labels.npy    int64   [N]
  <output_dir>/class_names.json
  <output_dir>/preprocess_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

# --- locate repo root so we can import the shared protocol -------------------
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]  # <root>/CNN/spectral-history-cnn/scripts/preprocess_sweep.py
sys.path.insert(0, str(_REPO_ROOT))

from experiments.unified_protocol import (  # noqa: E402
    CANONICAL_CLASSES,
    make_observed_patch,
    required_source_crop,
)

# CNN package imports (run with cwd = CNN/spectral-history-cnn, PYTHONPATH=.)
from src.processing.residuals import tv_residual  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLIT_SEED_OFFSETS = {"train": 0, "val": 100_000, "test": 200_000}


def log_rfft_spectrum(residual: np.ndarray, dc_sigma_bins: float) -> np.ndarray:
    """Generalized log-rFFT magnitude (any HxW), DC-suppressed, vertical fftshift."""
    height, width = residual.shape
    x = torch.from_numpy(residual.astype(np.float32)).unsqueeze(0)
    f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    f = torch.fft.fftshift(f, dim=-2)
    u = torch.fft.rfftfreq(width, d=1.0)
    v = torch.fft.fftshift(torch.fft.fftfreq(height, d=1.0))
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    r_bins = torch.sqrt((u_grid * width) ** 2 + (v_grid * height) ** 2 + 1e-12)
    dc_weight = 1.0 - torch.exp(-0.5 * (r_bins / float(dc_sigma_bins)) ** 2)
    f = f * dc_weight.unsqueeze(0)
    return torch.log1p(torch.abs(f)).numpy().astype(np.float32)


def build_class_split(task: dict) -> tuple[str, int, np.ndarray, np.ndarray]:
    split = task["split"]
    class_index = task["class_index"]
    class_name = task["class_name"]
    observed_size = task["observed_size"]
    sources = task["sources"]
    input_dir = Path(task["input_dir"])
    n_target = task["n_target"]
    seed = task["seed"]
    tv_weight = task["tv_weight"]
    tv_eps = task["tv_eps"]
    tv_max_iter = task["tv_max_iter"]
    dc_sigma_bins = task["dc_sigma_bins"]

    width_rfft = observed_size // 2 + 1
    spectra = np.empty((n_target, 1, observed_size, width_rfft), dtype=np.float16)
    labels = np.full((n_target,), class_index, dtype=np.int64)

    rng = np.random.default_rng(seed + SPLIT_SEED_OFFSETS[split] + class_index * 1009)
    crop_size = required_source_crop(class_name, observed_size)
    made = 0
    attempts = 0
    max_attempts = max(n_target * 200, 5000)
    while made < n_target:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Only made {made}/{n_target} for {split}:{class_name}@{observed_size} "
                f"(needs source >= {crop_size}px)."
            )
        attempts += 1
        name = sources[int(rng.integers(0, len(sources)))]
        path = input_dir / name
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                patch = make_observed_patch(img, class_name, observed_size, rng)
        except Exception:
            continue
        if patch is None:
            continue
        arr = np.asarray(patch, dtype=np.float32) / 255.0
        y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        residual = tv_residual(y.astype(np.float32), weight=tv_weight, eps=tv_eps, max_num_iter=tv_max_iter)
        spectra[made] = log_rfft_spectrum(residual, dc_sigma_bins).astype(np.float16)
        made += 1
    return split, class_index, spectra, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--observed-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--tv-weight", type=float, default=0.08)
    parser.add_argument("--tv-eps", type=float, default=1e-4)
    parser.add_argument("--tv-max-iter", type=int, default=30)
    parser.add_argument("--dc-sigma-bins", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    split_data = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    # support both CNN ({"splits": {...}}) and mask ({"train":[...]}) layouts
    if "splits" in split_data:
        split_lists = split_data["splits"]
    else:
        split_lists = {k: split_data[k] for k in ("train", "val", "test")}

    per_class = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for split in ("train", "val", "test"):
        for class_index, class_name in enumerate(CANONICAL_CLASSES):
            tasks.append(
                {
                    "split": split,
                    "class_index": class_index,
                    "class_name": class_name,
                    "observed_size": args.observed_size,
                    "sources": split_lists[split],
                    "input_dir": args.input_dir,
                    "n_target": per_class[split],
                    "seed": args.seed,
                    "tv_weight": args.tv_weight,
                    "tv_eps": args.tv_eps,
                    "tv_max_iter": args.tv_max_iter,
                    "dc_sigma_bins": args.dc_sigma_bins,
                }
            )

    results: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {s: {} for s in ("train", "val", "test")}
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        futures = [ex.submit(build_class_split, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"o={args.observed_size}", unit="task"):
            split, class_index, spectra, labels = fut.result()
            results[split][class_index] = (spectra, labels)

    for split in ("train", "val", "test"):
        chunks = [results[split][ci] for ci in range(len(CANONICAL_CLASSES))]
        spectra = np.concatenate([c[0] for c in chunks], axis=0)
        labels = np.concatenate([c[1] for c in chunks], axis=0)
        np.save(output_dir / f"{split}_spectra.npy", spectra)
        np.save(output_dir / f"{split}_labels.npy", labels)
        print(f"{split}: spectra={spectra.shape} labels={np.bincount(labels)}")

    (output_dir / "class_names.json").write_text(json.dumps(CANONICAL_CLASSES, indent=2), encoding="utf-8")
    (output_dir / "preprocess_config.json").write_text(
        json.dumps(
            {
                "observed_size": args.observed_size,
                "classes": CANONICAL_CLASSES,
                "width_rfft": args.observed_size // 2 + 1,
                "per_class": per_class,
                "tv_weight": args.tv_weight,
                "dc_sigma_bins": args.dc_sigma_bins,
                "seed": args.seed,
                "split_json": str(args.split_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done o={args.observed_size} -> {output_dir}")


if __name__ == "__main__":
    main()
