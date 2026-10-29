"""
baseline_trainer.py
===================
Task 2: Train and compare 7 baseline models.
Uses already-processed k-mer feature matrices in data/processed/.
Does NOT retrain or modify the existing DNABERT/GAT/XGBoost pipeline.

Baselines:
  1. Logistic Regression (on k-mer features)        → "Logistic Regression"
  2. Random Forest                                   → "Random Forest"
  3. SVM                                             → "SVM"
  4. K-Nearest Neighbours                            → "KNN"
  5. Naive Bayes                                     → "Naive Bayes"
  6. XGBoost (on k-mer features, no DL)             → "XGBoost (k-mer)"
  7. Proposed Hybrid (DNABERT+CNN+GAT+XGBoost)      → loaded from saved metrics
"""

import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from .metrics_engine import compute_all_metrics, metrics_table_rows

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


BASELINE_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "SVM": SVC(
        probability=True, kernel="rbf", random_state=RANDOM_SEED
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "XGBoost (k-mer)": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_SEED, eval_metric="logloss",
        verbosity=0, n_jobs=-1
    ),
}


def _load_splits(processed_dir: Path):
    """Load train / val / test splits from .npy files."""
    X_train = np.load(processed_dir / "train" / "X_train.npy")
    y_train = np.load(processed_dir / "train" / "y_train.npy")

    val_path = processed_dir / "val" / "X_val.npy"
    if val_path.exists():
        X_val = np.load(val_path)
        y_val = np.load(processed_dir / "val" / "y_val.npy")
    else:
        X_val, y_val = None, None

    X_test = np.load(processed_dir / "test" / "X_test.npy")
    y_test = np.load(processed_dir / "test" / "y_test.npy")

    logger.info(f"Splits loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_baselines(processed_dir: Path, hybrid_metrics_path: Path = None) -> List[Dict[str, Any]]:
    """
    Train all baseline models and return a list of metrics dicts.

    Args:
        processed_dir:       Path to data/processed/
        hybrid_metrics_path: Optional path to pre-computed hybrid metrics JSON.

    Returns:
        List of metric dicts, one per model.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = _load_splits(processed_dir)

    all_metrics: List[Dict[str, Any]] = []

    for name, clf in BASELINE_MODELS.items():
        logger.info(f"Training baseline: {name}")
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            inf_time = (time.time() - t1) / len(X_test)

            m = compute_all_metrics(
                y_test, y_pred, y_prob,
                model_name=name,
                training_time_s=train_time,
                inference_time_s=inf_time,
            )
            all_metrics.append(m)

        except Exception as e:
            logger.error(f"Baseline '{name}' failed: {e}")

    # Load pre-computed hybrid metrics if available
    if hybrid_metrics_path and Path(hybrid_metrics_path).exists():
        try:
            with open(hybrid_metrics_path) as f:
                hybrid = json.load(f)
            hybrid["model_name"] = "Proposed Hybrid (DNABERT+CNN+GAT+XGB)"
            all_metrics.append(hybrid)
            logger.info("Hybrid model metrics loaded from saved results.")
        except Exception as e:
            logger.warning(f"Could not load hybrid metrics: {e}")

    return all_metrics


def highlight_best(metrics_list: List[Dict]) -> Dict[str, str]:
    """Return a dict mapping metric name → best model name."""
    best = {}
    for metric in ["accuracy", "f1_score", "roc_auc", "mcc", "balanced_accuracy"]:
        best_model = max(metrics_list, key=lambda m: float(m.get(metric, 0)))
        best[metric] = best_model["model_name"]
    return best
