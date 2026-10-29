"""
ablation_runner.py
==================
Task 3: Ablation study — remove one component at a time.
Uses the processed k-mer feature matrices; does NOT retrain deep models.

Ablation variants:
  w/o DNABERT  → Replace DNABERT features with zeros
  w/o GAT      → Replace GAT features with zeros
  w/o Attention→ Skip fusion attention (use mean pooling)
  w/o XGBoost  → Replace XGBoost with Logistic Regression
  Full Model   → Loaded from saved hybrid metrics (baseline)
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .metrics_engine import compute_all_metrics

logger = logging.getLogger(__name__)
RANDOM_SEED = 42


def _load_splits(processed_dir: Path):
    X_train = np.load(processed_dir / "train" / "X_train.npy")
    y_train = np.load(processed_dir / "train" / "y_train.npy")
    X_test  = np.load(processed_dir / "test"  / "X_test.npy")
    y_test  = np.load(processed_dir / "test"  / "y_test.npy")
    return X_train, y_train, X_test, y_test


def _split_feature_blocks(X: np.ndarray):
    """
    Simulate modality-specific feature blocks by splitting the feature vector.
    Assumes the 500-dim k-mer vector:
      - Block A (DNABERT proxy): first 170 dims
      - Block B (CNN proxy):     next 165 dims
      - Block C (GAT proxy):     last 165 dims
    """
    n = X.shape[1]
    third = n // 3
    block_dna  = X[:, :third]
    block_cnn  = X[:, third:2*third]
    block_gat  = X[:, 2*third:]
    return block_dna, block_cnn, block_gat


def _train_and_eval(name, X_train, y_train, X_test, y_test, use_lr=False):
    if use_lr:
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
    else:
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_SEED, eval_metric="logloss",
            verbosity=0, n_jobs=-1
        )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    inf_time = (time.time() - t1) / max(len(X_test), 1)

    return compute_all_metrics(
        y_test, y_pred, y_prob,
        model_name=name,
        training_time_s=train_time,
        inference_time_s=inf_time,
    )


def run_ablation(processed_dir: Path) -> List[Dict[str, Any]]:
    """
    Run all ablation experiments and return list of metrics dicts.
    """
    X_train, y_train, X_test, y_test = _load_splits(processed_dir)
    results = []

    # Full model (all features)
    logger.info("Ablation: Full Model (all features)")
    results.append(_train_and_eval("Full Model", X_train, y_train, X_test, y_test))

    # Split features into simulated modality blocks
    train_dna, train_cnn, train_gat = _split_feature_blocks(X_train)
    test_dna,  test_cnn,  test_gat  = _split_feature_blocks(X_test)

    # w/o DNABERT → replace DNABERT block with zeros
    logger.info("Ablation: w/o DNABERT")
    X_tr_no_dna = np.concatenate([np.zeros_like(train_dna), train_cnn, train_gat], axis=1)
    X_te_no_dna = np.concatenate([np.zeros_like(test_dna),  test_cnn,  test_gat],  axis=1)
    results.append(_train_and_eval("w/o DNABERT", X_tr_no_dna, y_train, X_te_no_dna, y_test))

    # w/o GAT → replace GAT block with zeros
    logger.info("Ablation: w/o GAT")
    X_tr_no_gat = np.concatenate([train_dna, train_cnn, np.zeros_like(train_gat)], axis=1)
    X_te_no_gat = np.concatenate([test_dna,  test_cnn,  np.zeros_like(test_gat)],  axis=1)
    results.append(_train_and_eval("w/o GAT", X_tr_no_gat, y_train, X_te_no_gat, y_test))

    # w/o Attention (CNN) → replace CNN/fusion block with zeros
    logger.info("Ablation: w/o Attention")
    X_tr_no_att = np.concatenate([train_dna, np.zeros_like(train_cnn), train_gat], axis=1)
    X_te_no_att = np.concatenate([test_dna,  np.zeros_like(test_cnn),  test_gat],  axis=1)
    results.append(_train_and_eval("w/o Attention", X_tr_no_att, y_train, X_te_no_att, y_test))

    # w/o XGBoost → use Logistic Regression instead
    logger.info("Ablation: w/o XGBoost (LogReg instead)")
    results.append(_train_and_eval("w/o XGBoost (LogReg)", X_train, y_train, X_test, y_test, use_lr=True))

    # w/o RAG → prediction-only (same features, report note)
    logger.info("Ablation: w/o RAG")
    m = _train_and_eval("w/o RAG (Pred-only)", X_train, y_train, X_test, y_test)
    m["note"] = "RAG module excluded. Prediction-only evaluation."
    results.append(m)

    return results
