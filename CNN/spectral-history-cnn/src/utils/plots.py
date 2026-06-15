from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_train_curves(log_csv: str | Path, path: str | Path) -> None:
    log_csv = Path(log_csv)
    if not log_csv.exists():
        return
    df = pd.read_csv(log_csv)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["epoch"], df["train_acc"], label="train")
    axes[1].plot(df["epoch"], df["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrum_image(array: np.ndarray, path: str | Path, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array).squeeze()
    fig, ax = plt.subplots(figsize=(5, 7), dpi=150)
    im = ax.imshow(arr, cmap="magma", aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.set_xlabel("rFFT horizontal frequency bin")
    ax.set_ylabel("Vertical frequency bin")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_many_spectra_grid(
    arrays: Iterable[np.ndarray],
    path: str | Path,
    title: str | None = None,
    max_cols: int = 4,
) -> None:
    arrays = [np.asarray(a).squeeze() for a in arrays]
    if not arrays:
        return
    cols = min(max_cols, len(arrays))
    rows = int(np.ceil(len(arrays) / cols))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.6 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    for ax, arr in zip(axes, arrays):
        ax.imshow(arr, cmap="magma", aspect="auto", origin="lower")
        ax.axis("off")
    for ax in axes[len(arrays) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
