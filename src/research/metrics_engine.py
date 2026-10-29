"""
metrics_engine.py
=================
Tasks 1 & 4: Comprehensive evaluation metrics engine.
Computes 11 publication-quality metrics from model predictions.
Does NOT modify any ML model or prediction logic.

Metrics computed:
  Accuracy, Precision, Recall, Specificity, Sensitivity,
  F1 Score, ROC-AUC, PR-AUC, MCC, Balanced Accuracy, Cohen's Kappa
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Proposed Hybrid",
    training_time_s: float = 0.0,
    inference_time_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute all 11 research-grade evaluation metrics.

    Args:
        y_true:           Ground truth binary labels.
        y_pred:           Predicted binary labels.
        y_prob:           Predicted probabilities for positive class.
        model_name:       Name string for reporting.
        training_time_s:  Training duration in seconds.
        inference_time_s: Inference duration in seconds.

    Returns:
        Dictionary of all metrics, suitable for JSON serialisation.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity      = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # same as recall
    fpr, tpr, _      = roc_curve(y_true, y_prob)
    prec, rec, _     = precision_recall_curve(y_true, y_prob)

    metrics = {
        "model_name":        model_name,
        "accuracy":          round(accuracy_score(y_true, y_pred),               4),
        "precision":         round(precision_score(y_true, y_pred,               zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred,                  zero_division=0), 4),
        "sensitivity":       round(sensitivity,                                   4),
        "specificity":       round(specificity,                                   4),
        "f1_score":          round(f1_score(y_true, y_pred,                      zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob),                4),
        "pr_auc":            round(average_precision_score(y_true, y_prob),      4),
        "mcc":               round(matthews_corrcoef(y_true, y_pred),            4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred),      4),
        "cohen_kappa":       round(cohen_kappa_score(y_true, y_pred),            4),
        "confusion_matrix":  [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Healthy", "Infected"], output_dict=True
        ),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": prec.tolist(),
            "recall":    rec.tolist(),
        },
        "training_time_s":   round(training_time_s,   2),
        "inference_time_s":  round(inference_time_s,  4),
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1_score']:.4f} | AUC={metrics['roc_auc']:.4f} | "
        f"MCC={metrics['mcc']:.4f} | Kappa={metrics['cohen_kappa']:.4f}"
    )
    return metrics


def save_metrics(metrics: Dict[str, Any], output_dir: Path, filename: str = "full_metrics.json") -> Path:
    """Persist metrics dictionary to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    # Remove non-serialisable roc/pr curve data for the summary JSON
    summary = {k: v for k, v in metrics.items() if k not in ("roc_curve", "pr_curve")}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Metrics saved → {out_path}")
    return out_path


def metrics_table_rows(metrics_list) -> list:
    """Convert list of metrics dicts to a flat list-of-dicts table for CSV/Excel."""
    keys = [
        "model_name", "accuracy", "precision", "recall",
        "specificity", "f1_score", "roc_auc", "pr_auc",
        "mcc", "balanced_accuracy", "cohen_kappa",
        "training_time_s", "inference_time_s",
    ]
    return [{k: m.get(k, "") for k in keys} for m in metrics_list]
