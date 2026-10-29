"""
AVIAN DNA CLASSIFIER v2 - LIGHTWEIGHT CONCATENATION FUSION PIPELINE
======================================================================
Architecture Change:
  OLD: DNABERT-2 + CNN+BiLSTM + GAT → 4-Head Transformer Cross-Attention → XGBoost
  NEW: DNABERT-2 + CNN+BiLSTM + GAT → Simple Concat → ProjectionMLP → XGBoost

Key design decisions:
  - Same dataset, preprocessing, 70/15/15 split, random_state=42 as v1
  - FusionAttentionModel (cross-attention) replaced by LightweightConcatFusion
  - All three feature-extraction branches remain completely unchanged
  - Results saved for fair comparison against v1

Usage:
    python run_pipeline_v2.py
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# ── stdlib ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

# ── third-party ───────────────────────────────────────────────────────────────
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    matthews_corrcoef
)

# ── project imports ───────────────────────────────────────────────────────────
from src.preprocessing.dna_preprocessing import clean_dna, validate_dna
from src.representation.dnabert_encoder import DNABERTEncoder
from src.models.fusion_model import PyTorchCNNExtractor          # unchanged
from src.models.gat_mutation_model import GATMutationModel, sequence_to_graph  # unchanged
from src.models.lightweight_fusion import LightweightConcatFusion  # NEW
from src.models.xgboost_classifier import XGBoostClassificationHead

# ── output dirs ───────────────────────────────────────────────────────────────
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "research" / "reports"
XAI_DIR     = PROJECT_ROOT / "outputs" / "research" / "xai"
for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, XAI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m";  BOLD    = "\033[1m"
    GREEN   = "\033[92m"; YELLOW  = "\033[93m"
    RED     = "\033[91m"; CYAN    = "\033[96m"
    BLUE    = "\033[94m"; MAGENTA = "\033[95m"

def banner(text, color=C.CYAN):
    w = 66
    print(f"\n{color}{C.BOLD}+{'='*w}+")
    print(f"|  {text.ljust(w-2)}|")
    print(f"+{'='*w}+{C.RESET}")

def ok(m):   print(f"  {C.GREEN}[OK] {m}{C.RESET}")
def warn(m): print(f"  {C.YELLOW}[!!] {m}{C.RESET}")
def err(m):  print(f"  {C.RED}[XX] {m}{C.RESET}")
def info(m): print(f"       {m}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (identical to run_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_fasta_robust(file_path):
    records = []
    current_header = None
    current_seq = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith(";") or line_str.startswith("#"):
                    continue
                if line_str.startswith(">"):
                    if current_header is not None:
                        records.append((current_header, "".join(current_seq)))
                    current_header = line_str[1:]
                    current_seq = []
                else:
                    current_seq.append(line_str)
        if current_header is not None:
            records.append((current_header, "".join(current_seq)))
    except Exception:
        pass
    return records


def load_fasta_to_strings(directory_path, label, max_len=500):
    """Load ALL FASTA sequences from directory. Same logic as run_pipeline.py."""
    path = Path(directory_path)
    seqs, labels = [], []
    if not path.exists():
        return seqs, labels

    files = list(path.rglob("*.fasta")) + list(path.rglob("*.fa"))
    seen = set()
    label_name = "Healthy" if label == 0 else "Infected"
    info(f"Loading {label_name} sequences from {len(files)} files...")

    for file_path in files:
        for _, raw_seq in _parse_fasta_robust(file_path):
            cleaned = clean_dna(raw_seq)
            if not validate_dna(cleaned) or len(cleaned) < 10:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            seqs.append(cleaned[:max_len])
            labels.append(label)
    return seqs, labels


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (cross-attention replaced by concat fusion)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(sequences, dnabert, cnn_model, gat_model, fusion_model, device):
    """
    Extract fused feature vectors for a list of raw DNA strings.
    Pipeline: seq → DNABERT + CNN + GAT → LightweightConcatFusion → [1, 128]
    """
    fused_list = []
    n = len(sequences)
    fusion_model.eval()

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if i > 0 and i % 100 == 0:
                info(f"  {i}/{n} sequences processed...")

            # ── Branch 1: DNABERT-2 contextual embedding ──────────────────
            dna_emb = dnabert.encode(seq)            # [1, 768]

            # ── Branch 2: CNN spatial motif features ──────────────────────
            # Expand dnabert output along a dummy 'sequence' axis for Conv1D
            cnn_input = dna_emb.unsqueeze(2).expand(-1, -1, 50)  # [1, 768, 50]
            cnn_feat = cnn_model(cnn_input)                        # [1, 64]

            # ── Branch 3: GAT structural graph features ───────────────────
            graph = sequence_to_graph(dna_emb.expand(10, 768))
            gat_nodes, _ = gat_model(
                graph.x.to(device), graph.edge_index.to(device)
            )
            gat_feat = gat_nodes.mean(dim=0, keepdim=True)         # [1, 128]

            # ── Lightweight Concatenation Fusion (NEW) ────────────────────
            # No cross-attention; just concatenate and project via MLP
            fused = fusion_model(dna_emb, cnn_feat, gat_feat)      # [1, 128]

            fused_list.append(fused.detach().cpu().numpy())

    return np.vstack(fused_list)  # [n, 128]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def full_evaluation(y_true, y_pred, y_prob, split_name="Test"):
    """Compute and print all required metrics."""
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    mcc   = matthews_corrcoef(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["Healthy", "Infected"],
        zero_division=0
    )

    print(f"\n  {C.CYAN}{C.BOLD}--- {split_name} Metrics ---{C.RESET}")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"\n  Classification Report:")
    for line in report.strip().splitlines():
        print(f"    {line}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                auc=auc, mcc=mcc, report=report)


def save_confusion_matrix(y_true, y_pred, filename, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Healthy", "Infected"],
        yticklabels=["Healthy", "Infected"]
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    return cm.tolist()


def save_roc_curve(y_true, y_prob, filename, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"{label}  AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
            label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Lightweight Concat Fusion", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# XAI: SHAP
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(xgb_model, X_test, feature_names=None):
    """Run SHAP TreeExplainer on the XGBoost model."""
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(xgb_model.model)
        shap_values = explainer.shap_values(X_test)

        # Global importance bar chart
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:20]
        top_vals = mean_abs[top_idx]
        top_names = [f"feat_{i}" for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_names[::-1], top_vals[::-1], color="#06b6d4")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("SHAP Global Feature Importance (Top 20)", fontweight="bold")
        fig.tight_layout()
        shap_importance_path = XAI_DIR / "shap_importance.png"
        fig.savefig(shap_importance_path, dpi=150)
        plt.close(fig)

        # Summary beeswarm
        fig2 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False, max_display=20)
        shap_summary_path = XAI_DIR / "shap_summary.png"
        plt.savefig(shap_summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        # Waterfall for first sample
        fig3 = plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_test[0]
            ),
            show=False
        )
        shap_waterfall_path = XAI_DIR / "shap_waterfall.png"
        plt.savefig(shap_waterfall_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]
        try:
            base_val_float = float(base_val)
        except (ValueError, TypeError):
            # Fallback if SHAP returns a weird string like '[5E-1]'
            base_val_float = 0.5
            
        global_importance = [
            {"feature": f"feat_{i}", "value": float(mean_abs[i])}
            for i in top_idx
        ]
        result = {
            "shap_summary_png": str(shap_summary_path),
            "shap_importance_png": str(shap_importance_path),
            "shap_waterfall_png": str(shap_waterfall_path),
            "global_importance": global_importance,
            "shap_base_value": base_val_float,
            "shap_sample_idx": 0,
        }
        with open(XAI_DIR / "shap_results.json", "w") as f:
            json.dump(result, f, indent=2)
        ok("SHAP analysis complete.")
        return shap_values, result

    except Exception as e:
        warn(f"SHAP analysis failed: {e}")
        return None, {}


# ─────────────────────────────────────────────────────────────────────────────
# XAI: LIME
# ─────────────────────────────────────────────────────────────────────────────
def run_lime_analysis(xgb_model, X_train, X_test, y_test, n_samples=5):
    """Run LIME on a sample of test instances."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
        import matplotlib.pyplot as plt

        explainer = LimeTabularExplainer(
            X_train,
            feature_names=[f"feat_{i}" for i in range(X_train.shape[1])],
            class_names=["Healthy", "Infected"],
            mode="classification",
            random_state=RANDOM_SEED,
        )

        explanations = []
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for idx in range(min(n_samples, len(X_test))):
            exp = explainer.explain_instance(
                X_test[idx],
                xgb_model.model.predict_proba,
                num_features=10,
                labels=(1,)
            )
            top_features = [
                {"feature": f, "contribution": float(w)}
                for f, w in exp.as_list(label=1)
            ]
            prob = xgb_model.model.predict_proba(X_test[idx:idx+1])[0, 1]
            explanations.append({
                "sample_idx": idx,
                "true_label": "Infected" if y_test[idx] == 1 else "Healthy",
                "prob_infected": float(prob),
                "top_features": top_features,
            })

            # Bar plot for this sample
            names  = [f["feature"] for f in top_features]
            values = [f["contribution"] for f in top_features]
            colors = ["#10b981" if v > 0 else "#f87171" for v in values]
            axes[idx].barh(names, values, color=colors)
            axes[idx].set_title(
                f"Sample {idx} | True: {explanations[-1]['true_label']} | "
                f"P(Infected)={prob:.2%}", fontsize=9
            )
            axes[idx].axvline(0, color="black", linewidth=0.8)

        fig.suptitle("LIME Local Explanations", fontweight="bold")
        fig.tight_layout()
        lime_chart_path = XAI_DIR / "lime_explanation.png"
        fig.savefig(lime_chart_path, dpi=150)
        plt.close(fig)

        lime_result = {
            "lime_chart_png": str(lime_chart_path),
            "explanations": explanations,
        }
        with open(XAI_DIR / "lime_results.json", "w") as f:
            json.dump(lime_result, f, indent=2)
        ok("LIME analysis complete.")
        return lime_result

    except Exception as e:
        warn(f"LIME analysis failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# XAI: Perturbation Analysis + Fidelity
# ─────────────────────────────────────────────────────────────────────────────
def run_perturbation_fidelity(xgb_model, X_test, y_test, shap_values, n_samples=100):
    """
    Perturbation Analysis:
      Mask top-k SHAP features (set to 0) and measure probability drop.
    XAI Fidelity Test:
      Compare drop from XAI-targeted masking vs random masking.
    """
    if shap_values is None:
        warn("Skipping perturbation analysis – SHAP values not available.")
        return {}

    rng = np.random.RandomState(RANDOM_SEED)
    n = min(n_samples, len(X_test))
    top_k = 10

    xai_drops, random_drops = [], []

    for i in range(n):
        x = X_test[i:i+1].copy()
        orig_prob = float(xgb_model.model.predict_proba(x)[0, 1])

        # XAI-targeted mask (top-k SHAP features)
        top_idx_xai = np.argsort(np.abs(shap_values[i]))[::-1][:top_k]
        x_masked_xai = x.copy()
        x_masked_xai[0, top_idx_xai] = 0.0
        prob_xai = float(xgb_model.model.predict_proba(x_masked_xai)[0, 1])

        # Random mask (same number of features)
        rand_idx = rng.choice(x.shape[1], top_k, replace=False)
        x_masked_rand = x.copy()
        x_masked_rand[0, rand_idx] = 0.0
        prob_rand = float(xgb_model.model.predict_proba(x_masked_rand)[0, 1])

        xai_drops.append(orig_prob - prob_xai)
        random_drops.append(orig_prob - prob_rand)

    mean_xai_drop  = float(np.mean(xai_drops))
    mean_rand_drop = float(np.mean(random_drops))
    fidelity_score = mean_xai_drop - mean_rand_drop

    # Cohen's d
    pooled_std = float(np.sqrt(
        (np.std(xai_drops) ** 2 + np.std(random_drops) ** 2) / 2
    ))
    cohens_d = (mean_xai_drop - mean_rand_drop) / (pooled_std + 1e-8)

    print(f"\n  {C.CYAN}--- Perturbation & Fidelity Results ---{C.RESET}")
    print(f"  Mean prob drop (XAI-targeted) : {mean_xai_drop:.4f}")
    print(f"  Mean prob drop (random)        : {mean_rand_drop:.4f}")
    print(f"  XAI Fidelity Score             : {fidelity_score:+.4f}")
    print(f"  Cohen's d                      : {cohens_d:.4f}")

    result = {
        "mean_xai_drop": mean_xai_drop,
        "mean_random_drop": mean_rand_drop,
        "fidelity_score": fidelity_score,
        "cohens_d": cohens_d,
        "n_samples": n,
        "top_k_masked": top_k,
    }

    # Bar comparison chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["XAI-Targeted Masking", "Random Masking"],
        [mean_xai_drop, mean_rand_drop],
        color=["#06b6d4", "#f87171"], alpha=0.85
    )
    ax.set_ylabel("Mean Probability Drop")
    ax.set_title(
        f"Perturbation Analysis  (Fidelity={fidelity_score:+.4f}, "
        f"Cohen's d={cohens_d:.2f})",
        fontweight="bold"
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=10
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_analysis.png", dpi=150)
    plt.close(fig)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def load_v1_results():
    """Load previous cross-attention architecture results if they exist."""
    v1_path = REPORTS_DIR / "results.json"
    if v1_path.exists():
        try:
            with open(v1_path) as f:
                data = json.load(f)
            # Extract top-level metrics if present
            metrics = data.get("metrics", data.get("performance", {}))
            if metrics:
                return metrics
        except Exception:
            pass
    return None


def save_comparison_chart(v1_metrics, v2_metrics):
    """Save a side-by-side bar chart comparing the two architectures."""
    metric_keys  = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]

    v1_vals = [v1_metrics.get(k, 0) for k in metric_keys]
    v2_vals = [v2_metrics.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, v1_vals, w, label="v1: Cross-Attention",   color="#8b5cf6", alpha=0.85)
    ax.bar(x + w/2, v2_vals, w, label="v2: Concat Fusion (new)", color="#06b6d4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Architecture Comparison: Cross-Attention vs Lightweight Concat Fusion",
                 fontweight="bold")
    ax.legend()

    for i, (v1, v2) in enumerate(zip(v1_vals, v2_vals)):
        ax.text(i - w/2, v1 + 0.01, f"{v1:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, v2 + 0.01, f"{v2:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "architecture_comparison.png", dpi=150)
    plt.close(fig)
    ok("Comparison chart saved: outputs/figures/architecture_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    banner("AVIAN DNA CLASSIFIER v2 — LIGHTWEIGHT CONCAT FUSION", C.MAGENTA)
    print("  Architecture: DNABERT-2 + CNN+BiLSTM + GAT -> Concat -> MLP -> XGBoost")
    print("  Cross-Attention fusion layer has been removed.\n")

    # ── STAGE 1: Load data ────────────────────────────────────────────────────
    banner("STAGE 1: LOAD FASTA DATA", C.BLUE)

    h_seqs, h_labels = load_fasta_to_strings(PROJECT_ROOT / "data/raw/healthy", 0)
    i_seqs, i_labels = load_fasta_to_strings(PROJECT_ROOT / "data/raw/infected", 1)
    all_seqs   = h_seqs + i_seqs
    all_labels = h_labels + i_labels

    if not all_seqs:
        err("No FASTA data found in data/raw/healthy or data/raw/infected.")
        sys.exit(1)

    all_seqs, all_labels = sk_shuffle(all_seqs, all_labels, random_state=RANDOM_SEED)
    all_labels = np.array(all_labels)

    # 70 / 15 / 15 stratified split — identical to run_pipeline.py
    X_temp, X_test_seq, y_temp, y_test = train_test_split(
        all_seqs, all_labels, test_size=0.15,
        random_state=RANDOM_SEED, stratify=all_labels
    )
    X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15 / 0.85),
        random_state=RANDOM_SEED, stratify=y_temp
    )
    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    ok("Data loaded — no downsampling applied.")
    print(f"  Total    : {len(all_seqs)}")
    print(f"  Healthy  : {len(h_seqs)}")
    print(f"  Infected : {len(i_seqs)}")
    print(f"  Train    : {len(X_train_seq)}")
    print(f"  Val      : {len(X_val_seq)}")
    print(f"  Test     : {len(X_test_seq)}")

    # ── STAGE 2: Initialise models ────────────────────────────────────────────
    banner("STAGE 2: INITIALISE DL MODELS", C.BLUE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device}")

    dnabert  = DNABERTEncoder(device=device)
    cnn_m    = PyTorchCNNExtractor(in_channels=768, num_filters1=128, num_filters2=64).to(device)
    gat_m    = GATMutationModel(in_channels=768, hidden_channels=64, out_channels=128).to(device)

    # ── NEW: Lightweight Concat Fusion (replaces FusionAttentionModel) ────────
    fusion_m = LightweightConcatFusion(
        dnabert_dim=768, cnn_dim=64, gat_dim=128,
        hidden1=512, hidden2=256, out_dim=128,
        dropout1=0.3, dropout2=0.2
    ).to(device)

    ok(f"LightweightConcatFusion initialised.  Params: {fusion_m.parameter_count():,}")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_clf = XGBoostClassificationHead(
        n_estimators=200,
        scale_pos_weight=float(scale_pos_weight),
        random_state=RANDOM_SEED
    )
    info(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    # ── STAGE 3: Feature extraction ───────────────────────────────────────────
    banner("STAGE 3: EXTRACT DEEP FEATURES", C.BLUE)
    info("Training set...")
    X_train_f = extract_features(X_train_seq, dnabert, cnn_m, gat_m, fusion_m, device)
    info("Validation set...")
    X_val_f   = extract_features(X_val_seq,   dnabert, cnn_m, gat_m, fusion_m, device)
    info("Test set...")
    X_test_f  = extract_features(X_test_seq,  dnabert, cnn_m, gat_m, fusion_m, device)
    ok(f"Feature shapes — train:{X_train_f.shape}  val:{X_val_f.shape}  test:{X_test_f.shape}")

    # Save features for XAI re-use (avoids recomputation)
    np.save(MODELS_DIR / "X_test_features_v2.npy",  X_test_f)
    np.save(MODELS_DIR / "y_test_labels_v2.npy",    y_test)
    np.save(MODELS_DIR / "X_train_features_v2.npy", X_train_f)

    # ── STAGE 4: Train XGBoost ────────────────────────────────────────────────
    banner("STAGE 4: TRAIN XGBOOST", C.BLUE)
    xgb_clf.train(X_train_f, y_train, X_val=X_val_f, y_val=y_val)
    ok("XGBoost training complete.")

    # Save XGBoost model
    xgb_path = MODELS_DIR / "xgboost_v2.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_clf, f)
    ok(f"Model saved: {xgb_path}")

    # ── STAGE 5: Evaluation ───────────────────────────────────────────────────
    banner("STAGE 5: EVALUATION", C.BLUE)

    # Validation
    y_val_prob = xgb_clf.model.predict_proba(X_val_f)[:, 1]
    y_val_pred = (y_val_prob > 0.5).astype(int)
    val_metrics = full_evaluation(y_val, y_val_pred, y_val_prob, "Validation")

    # Test
    y_test_prob = xgb_clf.model.predict_proba(X_test_f)[:, 1]
    y_test_pred = (y_test_prob > 0.5).astype(int)
    test_metrics = full_evaluation(y_test, y_test_pred, y_test_prob, "Test")

    # Figures
    cm_data = save_confusion_matrix(
        y_test, y_test_pred,
        "confusion_matrix_v2.png",
        "Confusion Matrix — Lightweight Concat Fusion"
    )
    save_roc_curve(
        y_test, y_test_prob,
        "roc_curve_v2.png",
        "Lightweight Concat Fusion"
    )
    ok("Evaluation figures saved.")

    # ── STAGE 6: XAI ─────────────────────────────────────────────────────────
    banner("STAGE 6: XAI ANALYSIS", C.BLUE)

    shap_values, shap_result = run_shap_analysis(xgb_clf, X_test_f)
    lime_result  = run_lime_analysis(xgb_clf, X_train_f, X_test_f, y_test, n_samples=5)
    perturb_result = run_perturbation_fidelity(xgb_clf, X_test_f, y_test, shap_values)

    # ── STAGE 7: Comparison with v1 ───────────────────────────────────────────
    banner("STAGE 7: ARCHITECTURE COMPARISON", C.BLUE)
    v1_metrics = load_v1_results()
    if v1_metrics:
        ok("Found previous Cross-Attention results. Generating comparison chart.")
        save_comparison_chart(v1_metrics, test_metrics)
    else:
        warn("No v1 results.json found. Comparison chart not generated.")
        info("Run the original run_pipeline.py first to generate v1 metrics.")

    # ── STAGE 8: Save full results JSON ───────────────────────────────────────
    banner("STAGE 8: SAVE RESULTS", C.BLUE)

    results = {
        "architecture": "v2_lightweight_concat_fusion",
        "timestamp": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "dataset": {
            "total": len(all_seqs),
            "healthy": len(h_seqs),
            "infected": len(i_seqs),
            "train": len(X_train_seq),
            "val": len(X_val_seq),
            "test": len(X_test_seq),
        },
        "fusion_module": "LightweightConcatFusion",
        "fusion_params": fusion_m.parameter_count(),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm_data,
        "xai": {
            "shap": shap_result,
            "lime": lime_result,
            "perturbation_fidelity": perturb_result,
        },
        "figures": {
            "confusion_matrix": str(FIGURES_DIR / "confusion_matrix_v2.png"),
            "roc_curve": str(FIGURES_DIR / "roc_curve_v2.png"),
            "perturbation": str(FIGURES_DIR / "perturbation_analysis.png"),
            "comparison": str(FIGURES_DIR / "architecture_comparison.png"),
        }
    }

    results_path = REPORTS_DIR / "results_v2.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ok(f"Full results saved: {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE", C.GREEN)
    elapsed = round(time.time() - t0, 1)
    print(f"  Elapsed    : {elapsed}s")
    print(f"  Architecture : DNABERT-2 + CNN+BiLSTM + GAT → ConcatMLP → XGBoost")
    print(f"  Test Accuracy  : {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test ROC-AUC   : {test_metrics['auc']:.4f}")
    print(f"  Test MCC       : {test_metrics['mcc']:.4f}")
    print()


if __name__ == "__main__":
    main()
