from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_image(array: np.ndarray, path: str | Path, cmap: str = "gray") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_training_curves(log_csv: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(log_csv)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_acc"], label="train")
    plt.plot(df["epoch"], df["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curves.png", dpi=150)
    plt.close()
