from __future__ import annotations

from typing import Any

import numpy as np


def _manual_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def _per_class_report(cm: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    supports = []
    precisions = []
    recalls = []
    f1s = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        report[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": support,
        }
        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    total = max(sum(supports), 1)
    report["macro avg"] = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1s)) if f1s else 0.0,
        "support": int(sum(supports)),
    }
    report["weighted avg"] = {
        "precision": float(np.average(precisions, weights=supports)) if supports else 0.0,
        "recall": float(np.average(recalls, weights=supports)) if supports else 0.0,
        "f1-score": float(np.average(f1s, weights=supports)) if supports else 0.0,
        "support": int(sum(supports)),
    }
    return report


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = len(class_names)
    metrics: dict[str, Any] = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
    }

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        auc = {}
        for idx, name in enumerate(class_names):
            binary = (y_true == idx).astype(np.int32)
            if binary.min() == binary.max():
                auc[name] = None
            else:
                auc[name] = float(roc_auc_score(binary, probs[:, idx]))
        metrics["one_vs_rest_auc"] = auc
    except Exception as exc:
        cm = _manual_confusion_matrix(y_true, y_pred, num_classes)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classification_report"] = _per_class_report(cm, class_names)
        metrics["metrics_warning"] = f"sklearn metrics unavailable or failed: {exc}"
        auc = {}
        for idx, name in enumerate(class_names):
            binary = (y_true == idx).astype(np.int32)
            if binary.min() == binary.max():
                auc[name] = None
                continue
            try:
                from sklearn.metrics import roc_auc_score

                auc[name] = float(roc_auc_score(binary, probs[:, idx]))
            except Exception:
                auc[name] = None
        metrics["one_vs_rest_auc"] = auc

    return metrics
