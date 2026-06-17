import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import SpectraDataset
from src.evaluate import load_model
from src.utils.device import resolve_device, setup_device_env
from src.utils.io import ensure_dir, load_config
from src.utils.plots import save_image, save_training_curves


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def save_model_visuals(model, class_names: list[str], output_dir: Path) -> None:
    masks_dir = ensure_dir(output_dir / "figures" / "masks")
    refs_dir = ensure_dir(output_dir / "figures" / "references")
    masks = model.get_masks().detach().cpu().numpy()[:, 0]
    refs = model.get_references().detach().cpu().numpy()[:, 0]

    np.save(masks_dir / "masks.npy", masks)
    np.save(refs_dir / "references.npy", refs)

    for idx, name in enumerate(class_names):
        mask = masks[idx]
        ref = refs[idx] - refs[idx].mean()
        save_image(mask, masks_dir / f"mask_{idx}_{name}.png", cmap="gray")
        save_image(normalize_for_display(ref), refs_dir / f"reference_{idx}_{name}.png", cmap="gray")
        save_image(normalize_for_display(mask * ref), refs_dir / f"masked_reference_{idx}_{name}.png", cmap="gray")

    flat = masks.reshape(masks.shape[0], -1)
    overlap = flat @ flat.T
    denom = np.sqrt((flat**2).sum(axis=1, keepdims=True)) @ np.sqrt((flat**2).sum(axis=1, keepdims=True)).T
    overlap = overlap / (denom + 1e-12)
    np.save(output_dir / "figures" / "mask_overlap.npy", overlap)
    save_image(overlap, output_dir / "figures" / "mask_overlap.png", cmap="viridis")


def save_mean_spectra(config: dict, output_dir: Path) -> None:
    dataset = SpectraDataset(config["data_dir"], "train")
    sums = np.zeros((config["num_classes"], config["spectrum"]["height"], config["spectrum"]["width_rfft"]), dtype=np.float64)
    counts = np.zeros(config["num_classes"], dtype=np.int64)
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        label = int(y.item())
        sums[label] += x.numpy()[0]
        counts[label] += 1
    means = sums / counts[:, None, None]
    mean_dir = ensure_dir(output_dir / "figures" / "mean_spectra")
    np.save(mean_dir / "mean_spectra.npy", means.astype(np.float32))
    for idx, name in enumerate(config["class_names"]):
        save_image(normalize_for_display(means[idx]), mean_dir / f"mean_spectrum_{idx}_{name}.png", cmap="gray")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_device_env(config)
    device = resolve_device(config["training"]["device"])
    output_dir = Path(config["output_dir"])
    model = load_model(config, args.checkpoint, device)
    save_model_visuals(model, config["class_names"], output_dir)
    save_mean_spectra(config, output_dir)
    save_training_curves(output_dir / "logs" / "train_log.csv", output_dir / "figures")


if __name__ == "__main__":
    main()
