from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpectraDataset
from src.models.spectral_positional_cnn import SpectralPositionalCNN
from src.utils.device import get_device, setup_device_env, use_pin_memory
from src.utils.config_guard import reject_legacy_config_path
from src.utils.io import ensure_dir, load_yaml, save_json, update_nested
from src.utils.metrics import compute_classification_metrics
from src.utils.plots import save_confusion_matrix


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    update_nested(config, "paths", "processed_dir", args.processed_dir)
    update_nested(config, "paths", "output_dir", args.output_dir)
    update_nested(config, "training", "device", args.device)


def build_model(config: Dict[str, Any]) -> SpectralPositionalCNN:
    return SpectralPositionalCNN(
        num_classes=int(config["model"]["num_classes"]),
        height=int(config["spectrum"]["height"]),
        width=int(config["spectrum"]["width"]),
        lambdas=config["positional_encoding"]["lambdas"],
        axis_sigma=float(config["positional_encoding"]["axis_sigma"]),
        dropout=float(config["model"]["dropout"]),
    )


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_indices = []
    all_labels = []
    all_probs = []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
        all_indices.append(idx.numpy())
    return np.concatenate(all_indices), np.concatenate(all_labels), np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SpectralPositionalCNN.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    reject_legacy_config_path(args.config)
    config = load_yaml(args.config)
    apply_cli_overrides(config, args)
    setup_device_env(config)
    device = get_device(config)
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = ensure_dir(config["paths"]["output_dir"])
    figures_dir = ensure_dir(output_dir / "figures")
    dataset = SpectraDataset(processed_dir, args.split, load_metadata=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=use_pin_memory(device, config),
    )

    indices, y_true, probs = predict(model, loader, device)
    y_pred = probs.argmax(axis=1)
    class_names = list(checkpoint.get("class_names", config["data"]["classes"]))
    metrics = compute_classification_metrics(y_true, y_pred, probs, class_names)

    metrics_path = output_dir / f"metrics_{args.split}.json"
    save_json(metrics, metrics_path)
    if args.split == "test":
        save_json(metrics, output_dir / "metrics.json")

    cm = np.asarray(metrics["confusion_matrix"])
    save_confusion_matrix(cm, class_names, figures_dir / "confusion_matrix.png")

    pred_df = pd.DataFrame(
        {
            "index": indices,
            "true_label": y_true,
            "pred_label": y_pred,
        }
    )
    for i, name in enumerate(class_names):
        pred_df[f"prob_{name}"] = probs[:, i]

    wanted_meta_cols = ["source_filename", "crop_size", "crop_x", "crop_y", "jpeg_quality", "interpolation"]
    if dataset.metadata is not None:
        # SpectraDataset keeps metadata as a list[dict]; normalize to DataFrame for indexed join.
        meta_df = pd.DataFrame(dataset.metadata)
        meta = meta_df.iloc[indices].reset_index(drop=True)
        for col in wanted_meta_cols:
            pred_df[col] = meta[col].values if col in meta.columns else ""
    else:
        for col in wanted_meta_cols:
            pred_df[col] = ""

    pred_df.to_csv(output_dir / f"predictions_{args.split}.csv", index=False)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {output_dir / f'predictions_{args.split}.csv'}")


if __name__ == "__main__":
    main()
