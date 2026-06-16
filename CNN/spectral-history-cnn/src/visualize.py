from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpectraDataset
from src.models.spectral_positional_cnn import SpectralPositionalCNN
from src.utils.device import get_device, setup_device_env
from src.utils.io import ensure_dir, load_yaml, update_nested
from src.utils.plots import save_many_spectra_grid, save_spectrum_image


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


def save_mean_and_example_spectra(
    dataset: SpectraDataset,
    class_names: list[str],
    figures_dir: Path,
    examples_per_class: int,
) -> None:
    labels = dataset.labels
    spectra = dataset.spectra.astype(np.float32)
    mean_dir = ensure_dir(figures_dir / "mean_spectrum_per_class")
    example_dir = ensure_dir(figures_dir / "example_spectra_per_class")

    for class_id, class_name in enumerate(class_names):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) == 0:
            continue
        mean_spectrum = spectra[class_indices].mean(axis=0)
        save_spectrum_image(mean_spectrum, mean_dir / f"{class_id}_{class_name}.png", title=f"mean {class_name}")

        chosen = class_indices[:examples_per_class]
        examples = [spectra[i] for i in chosen]
        save_many_spectra_grid(examples, example_dir / f"{class_id}_{class_name}.png", title=f"examples {class_name}")


def save_gradient_saliency(
    model: torch.nn.Module,
    dataset: SpectraDataset,
    class_names: list[str],
    figures_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_per_class: int,
) -> None:
    saliency_dir = ensure_dir(figures_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    accum = {i: np.zeros((1, 64, 33), dtype=np.float64) for i in range(len(class_names))}
    counts = {i: 0 for i in range(len(class_names))}

    model.eval()
    for x, y, _ in loader:
        if all(counts[i] >= max_per_class for i in counts):
            break
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(x)
        preds = logits.argmax(dim=1)
        correct = preds.eq(y)
        if not bool(correct.any()):
            continue

        selected = logits[torch.arange(logits.shape[0], device=device), preds]
        loss = selected[correct].sum()
        loss.backward()

        grad = x.grad.detach().abs().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        correct_cpu = correct.detach().cpu().numpy()
        for i in range(len(y_cpu)):
            if not correct_cpu[i]:
                continue
            class_id = int(y_cpu[i])
            if counts[class_id] >= max_per_class:
                continue
            accum[class_id] += grad[i]
            counts[class_id] += 1

    for class_id, class_name in enumerate(class_names):
        if counts[class_id] == 0:
            print(f"No correctly classified samples found for saliency class {class_name}.")
            continue
        saliency = accum[class_id] / counts[class_id]
        save_spectrum_image(
            saliency,
            saliency_dir / f"saliency_{class_name}.png",
            title=f"saliency {class_name} n={counts[class_id]}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create spectrum and saliency visualizations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples-per-class", type=int, default=8)
    parser.add_argument("--saliency-per-class", type=int, default=128)
    args = parser.parse_args()

    config = load_yaml(args.config)
    apply_cli_overrides(config, args)
    setup_device_env(config)
    device = get_device(config)
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = list(checkpoint.get("class_names", config["data"]["classes"]))

    processed_dir = Path(config["paths"]["processed_dir"])
    figures_dir = ensure_dir(Path(config["paths"]["output_dir"]) / "figures")
    dataset = SpectraDataset(processed_dir, args.split)

    save_mean_and_example_spectra(dataset, class_names, figures_dir, args.examples_per_class)
    save_gradient_saliency(
        model,
        dataset,
        class_names,
        figures_dir,
        device,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        max_per_class=int(args.saliency_per_class),
    )
    print(f"Saved figures under {figures_dir}")


if __name__ == "__main__":
    main()
