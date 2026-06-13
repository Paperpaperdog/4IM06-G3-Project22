import argparse
import csv
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import SpectraDataset
from src.models.spectral_mask_classifier import SpectralMaskClassifier
from src.utils.io import ensure_dir, load_config
from src.utils.seed import set_seed


def make_loader(config: dict, split: str, shuffle: bool) -> DataLoader:
    training = config["training"]
    dataset = SpectraDataset(config["data_dir"], split)
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training["num_workers"],
        pin_memory=training["pin_memory"],
    )


def make_model(config: dict) -> SpectralMaskClassifier:
    spectrum = config["spectrum"]
    model_cfg = config["model"]
    return SpectralMaskClassifier(
        num_classes=config["num_classes"],
        height=spectrum["height"],
        width_rfft=spectrum["width_rfft"],
        init_mask_logits=model_cfg["init_mask_logits"],
        init_reference_std=model_cfg["init_reference_std"],
    )


def run_epoch(model, loader, device, optimizer=None, lambda_mask_l1: float = 0.0) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss = loss + lambda_mask_l1 * model.get_masks().mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * y.shape[0]
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.shape[0])

    return total_loss / total, total_correct / total


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, config: dict) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    output_dir = ensure_dir(config["output_dir"])
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")
    shutil.copyfile(args.config, output_dir / "config.yaml")

    device = torch.device(config["training"]["device"])
    train_loader = make_loader(config, "train", shuffle=True)
    val_loader = make_loader(config, "val", shuffle=False)
    model = make_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    lambda_mask_l1 = config["model"]["lambda_mask_l1"]
    best_val_loss = float("inf")
    log_path = output_dir / "logs" / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        writer.writeheader()

        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, lambda_mask_l1)
            val_loss, val_acc = run_epoch(model, val_loader, device, None, lambda_mask_l1)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
            }
            writer.writerow(row)
            f.flush()

            save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, row, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, epoch, row, config)

            print(
                f"epoch {epoch:03d} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}"
            )


if __name__ == "__main__":
    main()
