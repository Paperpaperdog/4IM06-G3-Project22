import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    auc = {}
    for idx, name in enumerate(class_names):
        binary = (y_true == idx).astype(np.int32)
        auc[name] = float(roc_auc_score(binary, probs[:, idx]))
    metrics["one_vs_rest_auc"] = auc
    return metrics
