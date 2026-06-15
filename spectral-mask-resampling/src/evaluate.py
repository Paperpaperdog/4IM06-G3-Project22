import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SpectraDataset
from src.models.spectral_mask_classifier import SpectralMaskClassifier
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.metrics import compute_metrics
from src.utils.plots import save_confusion_matrix


def load_model(config: dict, checkpoint_path: str | Path, device: torch.device) -> SpectralMaskClassifier:
    spectrum = config["spectrum"]
    model_cfg = config["model"]
    model = SpectralMaskClassifier(
        num_classes=config["num_classes"],
        height=spectrum["height"],
        width_rfft=spectrum["width_rfft"],
        init_mask_logits=model_cfg["init_mask_logits"],
        init_reference_std=model_cfg["init_reference_std"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["training"]["device"])
    output_dir = Path(config["output_dir"])
    figures_dir = ensure_dir(output_dir / "figures")
    dataset = SpectraDataset(config["data_dir"], args.split)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    model = load_model(config, args.checkpoint, device)

    all_true = []
    all_probs = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"evaluate {args.split}"):
            logits, _ = model(x.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    probs = np.concatenate(all_probs)
    y_pred = probs.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, probs, config["class_names"])
    observed_sizes_path = Path(config["data_dir"]) / f"{args.split}_observed_sizes.npy"
    observed_sizes = np.load(observed_sizes_path, mmap_mode="r")
    metrics["accuracy_by_observed_size"] = {}
    by_size_dir = ensure_dir(figures_dir / "confusion_matrix_by_observed_size")
    for observed_size in config["observed_sizes"]:
        mask = np.asarray(observed_sizes) == observed_size
        size_metrics = compute_metrics(y_true[mask], y_pred[mask], probs[mask], config["class_names"])
        metrics["accuracy_by_observed_size"][str(observed_size)] = size_metrics["accuracy"]
        save_confusion_matrix(
            np.asarray(size_metrics["confusion_matrix"]),
            config["class_names"],
            by_size_dir / f"confusion_matrix_observed_size_{observed_size}.png",
        )
    save_json(metrics, output_dir / "metrics.json")
    save_confusion_matrix(np.asarray(metrics["confusion_matrix"]), config["class_names"], figures_dir / "confusion_matrix.png")

    rows = {
        "index": np.arange(len(y_true)),
        "true_label": y_true,
        "pred_label": y_pred,
        "observed_size": np.asarray(observed_sizes),
    }
    for idx, name in enumerate(config["class_names"]):
        rows[f"prob_{name}"] = probs[:, idx]
    pd.DataFrame(rows).to_csv(output_dir / f"predictions_{args.split}.csv", index=False)
    print(f"{args.split}_accuracy={metrics['accuracy']:.6f}")


if __name__ == "__main__":
    main()
