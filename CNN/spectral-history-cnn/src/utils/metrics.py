from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _manual_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray],
    class_names: List[str],
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = len(class_names)
    out: Dict[str, Any] = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
    }

    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

        labels = list(range(num_classes))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["per_class"] = {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        }
        out["confusion_matrix"] = cm.tolist()

        if probs is not None:
            auc = {}
            for i, name in enumerate(class_names):
                binary_true = (y_true == i).astype(np.int64)
                if binary_true.min() == binary_true.max():
                    auc[name] = None
                else:
                    auc[name] = float(roc_auc_score(binary_true, probs[:, i]))
            out["one_vs_rest_auc"] = auc
    except Exception as exc:  # pragma: no cover - fallback for minimal installs.
        cm = _manual_confusion_matrix(y_true, y_pred, num_classes)
        out["confusion_matrix"] = cm.tolist()
        out["metrics_warning"] = f"sklearn metrics unavailable or failed: {exc}"
        per_class = {}
        for i, name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(cm[i, :].sum()),
            }
        out["per_class"] = per_class

    confusion_pairs = [
        ("JPEG", "downsample_x8"),
        ("JPEG", "downsample_x16"),
        ("downsample_x4", "downsample_x8"),
        ("downsample_x8", "downsample_x16"),
        ("original", "JPEG"),
        ("original", "downsample_x2"),
    ]
    pair_counts = {}
    cm = np.asarray(out["confusion_matrix"])
    name_to_id = {name: i for i, name in enumerate(class_names)}
    for a, b in confusion_pairs:
        if a in name_to_id and b in name_to_id:
            ia, ib = name_to_id[a], name_to_id[b]
            pair_counts[f"{a}_as_{b}"] = int(cm[ia, ib])
            pair_counts[f"{b}_as_{a}"] = int(cm[ib, ia])
    out["important_confusion_pairs"] = pair_counts
    return out
