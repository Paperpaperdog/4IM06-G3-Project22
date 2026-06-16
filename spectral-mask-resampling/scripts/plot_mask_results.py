#!/usr/bin/env python3
"""Generate summary figures for v1_fourier_ambiguity_mask_clean results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CLASS_NAMES = ["original", "JPEG_Q80", "downsample×8", "downsample×16"]
CLASS_NAMES_SHORT = ["original", "JPEG", "×8", "×16"]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_per_class_metrics(metrics: dict, out_dir: Path) -> None:
    report = metrics["classification_report"]
    auc = metrics["one_vs_rest_auc"]
    classes = [c for c in CLASS_NAMES if c.replace("×", "x").replace("downsample", "downsample") in report or True]
    # use keys from metrics
    keys = ["original", "JPEG_Q80", "downsample_x8", "downsample_x16"]
    labels = CLASS_NAMES
    f1 = [report[k]["f1-score"] for k in keys]
    auc_vals = [auc[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, f1, width, label="F1", color="#4C72B0")
    ax.bar(x + width / 2, auc_vals, width, label="AUC (OvR)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class F1 and AUC (test set)")
    ax.axhline(0.25, color="gray", ls="--", lw=1, alpha=0.6, label="random (4-class)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_metrics.png", dpi=160)
    plt.close(fig)


def plot_accuracy_by_size(metrics: dict, out_dir: Path) -> None:
    sizes = sorted(int(k) for k in metrics["accuracy_by_observed_size"])
    accs = [metrics["accuracy_by_observed_size"][str(s)] for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(s) for s in sizes], [a * 100 for a in accs], color="#C44E52", alpha=0.85)
    ax.axhline(56.6, color="#4C72B0", ls="--", lw=1.5, label="overall 56.6%")
    ax.axhline(25, color="gray", ls=":", lw=1, alpha=0.7, label="random 25%")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Observed patch size (px)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test accuracy vs observed patch size")
    ax.set_ylim(0, 75)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_size.png", dpi=160)
    plt.close(fig)


def plot_normalized_confusion(metrics: dict, out_dir: Path) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES_SHORT,
        yticklabels=CLASS_NAMES_SHORT,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "row-normalized proportion"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (row-wise)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_normalized.png", dpi=160)
    plt.close(fig)


def plot_key_confusion_pairs(metrics: dict, out_dir: Path) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    keys = ["original", "JPEG_Q80", "downsample_x8", "downsample_x16"]
    pairs = [
        ("original", "JPEG_Q80", 0, 1),
        ("JPEG_Q80", "original", 1, 0),
        ("JPEG_Q80", "downsample×8", 1, 2),
        ("JPEG_Q80", "downsample×16", 1, 3),
        ("downsample×8", "downsample×16", 2, 3),
        ("downsample×16", "downsample×8", 3, 2),
        ("original", "downsample×8", 0, 2),
        ("original", "downsample×16", 0, 3),
    ]
    labels = [f"{a} → {b}" for a, b, _, _ in pairs]
    counts = [cm[i, j] for _, _, i, j in pairs]

    order = np.argsort(counts)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52" if "×8" in l and "×16" in l else "#8172B2" for l in labels]
    ax.barh(labels, counts, color=colors)
    ax.set_xlabel("Misclassification count (support=5000 per class)")
    ax.set_title("Key off-diagonal confusion pairs")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "key_confusion_pairs.png", dpi=160)
    plt.close(fig)


def plot_mask_overlap(overlap: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        overlap,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=CLASS_NAMES_SHORT,
        yticklabels=CLASS_NAMES_SHORT,
        vmin=0.85,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "cosine overlap"},
    )
    offdiag = overlap[~np.eye(overlap.shape[0], dtype=bool)]
    ax.set_title(f"Learned mask overlap (off-diag mean={offdiag.mean():.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "mask_overlap_heatmap.png", dpi=160)
    plt.close(fig)


def plot_learned_masks(masks: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name, mask in zip(axes.ravel(), CLASS_NAMES, masks):
        im = ax.imshow(mask, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(f"Learned mask: {name}")
        ax.set_xlabel("horizontal freq bin")
        ax.set_ylabel("vertical freq bin")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Per-class learned spectral masks (sigmoid)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "learned_masks.png", dpi=160)
    plt.close(fig)


def plot_mean_spectra(spectra: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    vmax = float(np.percentile(spectra, 99))
    for ax, name, spec in zip(axes.ravel(), CLASS_NAMES, spectra):
        im = ax.imshow(spec, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(f"Mean log-spectrum: {name}")
        ax.set_xlabel("horizontal freq bin")
        ax.set_ylabel("vertical freq bin")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Class-conditional mean spectra (test set)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_spectra.png", dpi=160)
    plt.close(fig)


def plot_prob_calibration_sample(predictions_csv: Path, out_dir: Path, n_per_class: int = 400) -> None:
    df = pd.read_csv(predictions_csv)
    label_map = {0: "original", 1: "JPEG_Q80", 2: "downsample_x8", 3: "downsample_x16"}
    df["true_name"] = df["true_label"].map(label_map)
    prob_cols = ["prob_original", "prob_JPEG_Q80", "prob_downsample_x8", "prob_downsample_x16"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, cls_id, name in zip(axes.ravel(), range(4), CLASS_NAMES):
        sub = df[df["true_label"] == cls_id]
        if len(sub) > n_per_class:
            sub = sub.sample(n_per_class, random_state=42)
        data = [sub[c].values for c in prob_cols]
        ax.boxplot(data, tick_labels=CLASS_NAMES_SHORT)
        ax.set_title(f"True class: {name}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
    fig.supxlabel("Predicted class probability")
    fig.supylabel("Probability")
    fig.suptitle("Probability distribution by true class (subsampled)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "prob_distribution_by_true_class.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/v1_fourier_ambiguity_mask_clean"),
    )
    args = parser.parse_args()

    out_root = args.output_dir
    fig_dir = _ensure_dir(out_root / "figures" / "summary")

    metrics = json.loads((out_root / "metrics.json").read_text())
    masks = np.load(out_root / "figures" / "masks" / "masks.npy")
    overlap = np.load(out_root / "figures" / "mask_overlap.npy")
    mean_spectra = np.load(out_root / "figures" / "mean_spectra" / "mean_spectra.npy")

    sns.set_theme(style="whitegrid", font_scale=1.05)
    plot_per_class_metrics(metrics, fig_dir)
    plot_accuracy_by_size(metrics, fig_dir)
    plot_normalized_confusion(metrics, fig_dir)
    plot_key_confusion_pairs(metrics, fig_dir)
    plot_mask_overlap(overlap, fig_dir)
    plot_learned_masks(masks, fig_dir)
    plot_mean_spectra(mean_spectra, fig_dir)
    plot_prob_calibration_sample(out_root / "predictions_test.csv", fig_dir)
    print(f"Saved summary figures to {fig_dir}")


if __name__ == "__main__":
    main()
