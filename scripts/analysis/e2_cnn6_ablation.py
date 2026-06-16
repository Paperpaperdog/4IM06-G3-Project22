#!/usr/bin/env python3
"""E2: Ablate 6-class CNN metrics to 4-class subset (no retraining)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "CNN/spectral-history-cnn/outputs/v1_final64_poscnn/metrics_test.json"
OUT_DIR = ROOT / "docs/tables"
OUT_PATH = OUT_DIR / "e2_cnn6_ablation.md"

CLASS_NAMES_6 = [
    "original",
    "JPEG",
    "downsample_x2",
    "downsample_x4",
    "downsample_x8",
    "downsample_x16",
]
IDX_4 = [0, 1, 4, 5]  # original, JPEG, x8, x16
NAMES_4 = [CLASS_NAMES_6[i] for i in IDX_4]


def subset_confusion(cm: list[list[int]], indices: list[int]) -> np.ndarray:
    arr = np.array(cm, dtype=np.int64)
    return arr[np.ix_(indices, indices)]


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def per_class_f1(cm: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = {"precision": prec, "recall": rec, "f1": f1, "support": int(cm[i, :].sum())}
    return out


def binary_acc(cm6: np.ndarray, i: int, j: int) -> float:
    sub = cm6[np.ix_([i, j], [i, j])]
    return accuracy_from_cm(sub)


def main() -> None:
    if not METRICS_PATH.exists():
        raise SystemExit(f"Missing metrics: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text())
    cm6 = np.array(metrics["confusion_matrix"], dtype=np.int64)
    cm4 = subset_confusion(metrics["confusion_matrix"], IDX_4)

    acc6 = metrics["accuracy"]
    acc4 = accuracy_from_cm(cm4)
    f1_4 = per_class_f1(cm4, NAMES_4)
    macro_f1_4 = float(np.mean([v["f1"] for v in f1_4.values()]))

    x8_i, x16_i = 4, 5
    x8_x16_acc = binary_acc(cm6, x8_i, x16_i)
    bridge_x8_to_x4 = int(cm6[x8_i, 3])
    bridge_x4_to_x8 = int(cm6[3, x8_i])
    confuse_8_16 = int(cm6[x8_i, x16_i])
    confuse_16_8 = int(cm6[x16_i, x8_i])

    pairs = metrics.get("important_confusion_pairs", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E2: 6-class CNN ablation (from existing metrics)\n",
        f"Source: `{METRICS_PATH.relative_to(ROOT)}`\n",
        "## Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| 6-class accuracy | {acc6:.1%} |",
        f"| 4-class subset accuracy | {acc4:.1%} |",
        f"| 4-class macro F1 | {macro_f1_4:.3f} |",
        f"| ×8/×16 binary accuracy (true ∈ {{×8,×16}}) | {x8_x16_acc:.1%} |",
        f"| ×8→×16 (6-class) | {confuse_8_16} |",
        f"| ×16→×8 (6-class) | {confuse_16_8} |",
        f"| ×8→×4 (bridge) | {bridge_x8_to_x4} |",
        f"| ×4→×8 (bridge) | {bridge_x4_to_x8} |",
        "",
        "## 4-class subset F1\n",
        "| Class | F1 | Recall | Support |",
        "|-------|-----|--------|---------|",
    ]
    for name in NAMES_4:
        s = f1_4[name]
        lines.append(f"| {name} | {s['f1']:.3f} | {s['recall']:.1%} | {s['support']} |")

    lines.extend(
        [
            "",
            "## 4-class confusion matrix\n",
            "```",
            str(cm4.tolist()),
            "```",
            "",
            "## Key pairs (from metrics JSON)\n",
        ]
    )
    for k, v in sorted(pairs.items()):
        lines.append(f"- {k}: {v}")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")
    print(f"6-class acc: {acc6:.1%} | 4-class subset acc: {acc4:.1%} | ×8/×16 binary: {x8_x16_acc:.1%}")


if __name__ == "__main__":
    main()
