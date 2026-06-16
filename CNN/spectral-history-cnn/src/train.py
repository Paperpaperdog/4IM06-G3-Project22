from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SpectraDataset
from src.models.spectral_positional_cnn import SpectralPositionalCNN
from src.utils.device import (
    autocast_context,
    create_grad_scaler,
    get_device,
    setup_device_env,
    supports_amp,
    use_pin_memory,
)
from src.utils.io import ensure_dir, load_yaml, update_nested
from src.utils.plots import save_train_curves
from src.utils.seed import set_seed


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    update_nested(config, "paths", "processed_dir", args.processed_dir)
    update_nested(config, "paths", "output_dir", args.output_dir)
    update_nested(config, "training", "device", args.device)
    update_nested(config, "training", "epochs", args.epochs)
    update_nested(config, "training", "batch_size", args.batch_size)
    update_nested(config, "training", "num_workers", args.num_workers)
    update_nested(config, "training", "lr", args.lr)


def build_model(config: Dict[str, Any]) -> SpectralPositionalCNN:
    class_names = config["data"]["classes"]
    model = SpectralPositionalCNN(
        num_classes=int(config["model"].get("num_classes", len(class_names))),
        height=int(config["spectrum"]["height"]),
        width=int(config["spectrum"]["width"]),
        lambdas=config["positional_encoding"]["lambdas"],
        axis_sigma=float(config["positional_encoding"]["axis_sigma"]),
        dropout=float(config["model"]["dropout"]),
    )
    expected_channels = int(config["model"]["input_channels"])
    if model.input_channels != expected_channels:
        raise ValueError(f"Model input channels are {model.input_channels}, config expects {expected_channels}.")
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context(device, use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += batch_size

    return total_loss / max(total, 1), total_correct / max(total, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, Any],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_val_loss": best_val_loss,
            "class_names": list(config["data"]["classes"]),
            "model_input_channels": getattr(model, "input_channels", None),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpectralPositionalCNN.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path (e.g. checkpoints/last.pt).")
    args = parser.parse_args()

    config = load_yaml(args.config)
    apply_cli_overrides(config, args)

    setup_device_env(config)

    seed = int(config["training"].get("seed", config["data"]["seed"]))
    set_seed(seed)
    device = get_device(config)
    print(f"Using device: {device}")

    processed_dir = Path(config["paths"]["processed_dir"])
    output_dir = ensure_dir(config["paths"]["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    figures_dir = ensure_dir(output_dir / "figures")

    train_dataset = SpectraDataset(processed_dir, "train")
    val_dataset = SpectraDataset(processed_dir, "val")
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    pin_memory = use_pin_memory(device, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(config).to(device)
    print(f"Model input channels: {model.input_channels}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = None
    if str(config["training"].get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))
    use_amp = supports_amp(device, config)
    scaler = create_grad_scaler(device, use_amp)

    log_path = output_dir / "train_log.csv"
    start_epoch = 1
    best_metric = float("inf")
    best_val_loss = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
        log_mode = "a" if log_path.exists() else "w"
    else:
        log_mode = "w"

    with log_path.open(log_mode, newline="", encoding="utf-8") as f:
        if log_mode == "w":
            writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writeheader()

    epochs = int(config["training"]["epochs"])
    save_best_by = str(config["training"].get("save_best_by", "val_loss"))

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        print(
            f"epoch {epoch:03d}/{epochs} "
            f"lr={lr:.3e} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        best_val_loss = min(best_val_loss, val_loss)
        current_metric = val_loss if save_best_by == "val_loss" else -val_acc
        if current_metric < best_metric:
            best_metric = current_metric
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, config, best_val_loss)

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, best_val_loss)
        save_train_curves(log_path, figures_dir / "train_curves.png")


if __name__ == "__main__":
    main()
