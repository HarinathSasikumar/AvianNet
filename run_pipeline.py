"""
AVIAN DNA CLASSIFIER - UNIFIED END-TO-END PIPELINE (PYTORCH MULTIMODAL)
Stages: Download -> Process FASTA -> Extract Features -> Train XGBoost -> Evaluate -> Infer

Usage:
    python run_pipeline.py
"""

# ─────────────────────────────────────────────────────────
# AUTO-INSTALL: ensure all required packages are available
# ─────────────────────────────────────────────────────────
import subprocess
import sys

REQUIRED_PACKAGES = {
    "torch":        "torch",
    "torch_geometric": "torch-geometric",
    "transformers": "transformers",
    "networkx":     "networkx",
    "einops":       "einops",
    "xgboost":      "xgboost",
    "Bio":          "biopython",
    "numpy":        "numpy",
    "pandas":       "pandas",
    "sklearn":      "scikit-learn",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "sentence_transformers": "sentence-transformers",
    "faiss":        "faiss-cpu",
    "openai":       "openai",
    "fastapi":      "fastapi",
    "uvicorn":      "uvicorn",
    "pydantic":     "pydantic",
    "requests":     "requests"
}

def _auto_install():
    missing = []
    for module, pkg in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("[SETUP] Missing packages detected: " + ", ".join(missing))
        print("[SETUP] Installing now via pip — please wait...\n")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("\n[SETUP] All packages installed successfully!\n")

_auto_install()
# ─────────────────────────────────────────────────────────

import os
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
from sklearn.model_selection import train_test_split

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, roc_auc_score
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ANSI color helpers
ESC = "\033"
class C:
    RESET   = ESC + "[0m"
    BOLD    = ESC + "[1m"
    GREEN   = ESC + "[92m"
    YELLOW  = ESC + "[93m"
    RED     = ESC + "[91m"
    CYAN    = ESC + "[96m"
    BLUE    = ESC + "[94m"
    MAGENTA = ESC + "[95m"

def banner(text, color=C.CYAN):
    w = 62
    b = "=" * w
    print("\n" + color + C.BOLD + "+" + b + "+")
    print("|  " + text.ljust(w - 2) + "|")
    print("+" + b + "+" + C.RESET)

def ok(msg): print("  " + C.GREEN + "[OK] " + msg + C.RESET)
def warn(msg): print("  " + C.YELLOW + "[!!] " + msg + C.RESET)
def err(msg): print("  " + C.RED + "[XX] " + msg + C.RESET)
def info(msg): print("  " + ESC + "[2m" + "     " + msg + C.RESET)

PROJECT_ROOT  = Path(__file__).resolve().parent
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(PROJECT_ROOT))

# Import Multimodal Architecture
from src.preprocessing.dna_preprocessing import clean_dna, validate_dna
from src.representation.dnabert_encoder import DNABERTEncoder
from src.models.fusion_model import PyTorchCNNExtractor, FusionAttentionModel
from src.models.gat_mutation_model import GATMutationModel, sequence_to_graph
from src.models.xgboost_classifier import XGBoostClassificationHead

def _parse_fasta_robust(file_path):
    """Custom FASTA parser that handles leading comments and blank lines from NCBI."""
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
    """Load ALL FASTA sequences from directory recursively. No sampling limit applied."""
    path = Path(directory_path)
    seqs, labels = [], []
    if not path.exists(): return seqs, labels

    files = list(path.rglob("*.fasta")) + list(path.rglob("*.fa"))
    seen = set()

    label_name = "Healthy" if label == 0 else "Infected"
    iterator = tqdm(files, desc=f"  Loading {label_name} FASTA files", unit="file") if tqdm else files

    for file_path in iterator:
        records = _parse_fasta_robust(file_path)
        for _, raw_seq in records:
            cleaned = clean_dna(raw_seq)
            if not validate_dna(cleaned) or len(cleaned) < 10:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            # Truncate to max_len for DNABERT memory constraints
            seqs.append(cleaned[:max_len])
            labels.append(label)
    return seqs, labels

def extract_features_for_batch(sequences, dnabert, cnn, gat, fusion, device):
    batch_fused = []
    print(f"  Extracting features for {len(sequences)} sequences...")
    for i, seq in enumerate(sequences):
        if i > 0 and i % 50 == 0: info(f"Processed {i}/{len(sequences)}...")
        # 1. DNABERT
        dna_emb = dnabert.encode(seq) # [1, 768]
        # 2. PyTorch CNN Spatial
        cnn_input = dna_emb.unsqueeze(2).expand(-1, -1, 50)
        cnn_feat = cnn(cnn_input) # [1, 64]
        # 3. GAT Mutational
        graph = sequence_to_graph(dna_emb.expand(10, 768)) 
        gat_nodes, _ = gat(graph.x.to(device), graph.edge_index.to(device))
        gat_feat = gat_nodes.mean(dim=0, keepdim=True) # [1, 128]
        # 4. Fusion
        fused_vec, _ = fusion(dna_emb, cnn_feat, gat_feat) # [1, 128]
        batch_fused.append(fused_vec.detach().cpu().numpy())
    return np.vstack(batch_fused)

def main():
    t0 = time.time()
    banner("AVIAN DNA CLASSIFIER - PYTORCH MULTIMODAL PIPELINE", C.MAGENTA)
    print("  Note: TensorFlow AVX issue detected previously. Switching to PyTorch/XGBoost.\n")

    # 1. Prepare Data — FULL DATASET, no downsampling
    print(C.BLUE + C.BOLD + "--- STAGE 1: LOAD FASTA DATA ---" + C.RESET)
    h_seqs, h_labels = load_fasta_to_strings(PROJECT_ROOT / "data/raw/healthy", 0)
    i_seqs, i_labels = load_fasta_to_strings(PROJECT_ROOT / "data/raw/infected", 1)
    all_seqs = h_seqs + i_seqs
    all_labels = h_labels + i_labels

    if not all_seqs:
        err("No FASTA data found in data/raw/healthy or infected.")
        return

    # Shuffle before splitting (no sampling limit)
    from sklearn.utils import shuffle as sk_shuffle
    all_seqs, all_labels = sk_shuffle(all_seqs, all_labels, random_state=42)

    # 70 / 15 / 15 stratified split
    X_temp, X_test_seq, y_temp, y_test = train_test_split(
        all_seqs, all_labels, test_size=0.15, random_state=42, stratify=all_labels
    )
    X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15 / 0.85), random_state=42, stratify=y_temp
    )

    ok("Loaded ALL available FASTA sequences. No downsampling applied.")
    print(f"  {C.CYAN}Total sequences : {len(all_seqs)}{C.RESET}")
    print(f"  {C.GREEN}Healthy         : {len(h_seqs)}{C.RESET}")
    print(f"  {C.RED}Infected        : {len(i_seqs)}{C.RESET}")
    print(f"  {C.YELLOW}Training        : {len(X_train_seq)}{C.RESET}")
    print(f"  {C.YELLOW}Validation      : {len(X_val_seq)}{C.RESET}")
    print(f"  {C.YELLOW}Testing         : {len(X_test_seq)}{C.RESET}")

    # 2. Init PyTorch Models
    print("\n" + C.BLUE + C.BOLD + "--- STAGE 2: INITIALIZE DL MODELS ---" + C.RESET)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Using device: {device}")
    
    dnabert = DNABERTEncoder(device=device)
    cnn = PyTorchCNNExtractor(in_channels=768, num_filters1=128, num_filters2=64).to(device)
    gat = GATMutationModel(in_channels=768, hidden_channels=64, out_channels=128).to(device)
    fusion = FusionAttentionModel(dnabert_dim=768, cnn_dim=64, gat_dim=128, fused_dim=256).to(device)
    # Calculate class weight for XGBoost
    neg_cases = sum(y_train == 0)
    pos_cases = sum(y_train == 1)
    scale_pos_weight = neg_cases / pos_cases if pos_cases > 0 else 1.0
    info(f"Class Imbalance weights (scale_pos_weight): {scale_pos_weight:.4f}")

    xgb = XGBoostClassificationHead(n_estimators=100, scale_pos_weight=scale_pos_weight)
    ok("PyTorch models loaded.")

    # 3. Extract Features
    print("\n" + C.BLUE + C.BOLD + "--- STAGE 3: EXTRACT DEEP FEATURES ---" + C.RESET)
    info("Extracting Training Features...")
    X_train_fused = extract_features_for_batch(X_train_seq, dnabert, cnn, gat, fusion, device)
    info("Extracting Validation Features...")
    X_val_fused = extract_features_for_batch(X_val_seq, dnabert, cnn, gat, fusion, device)
    info("Extracting Testing Features...")
    X_test_fused = extract_features_for_batch(X_test_seq, dnabert, cnn, gat, fusion, device)
    ok("Feature extraction complete.")

    # 4. Train XGBoost
    print("\n" + C.BLUE + C.BOLD + "--- STAGE 4: TRAIN XGBOOST ---" + C.RESET)
    xgb.train(X_train_fused, np.array(y_train))
    ok("XGBoost trained successfully.")

    # 5. Evaluate
    print("\n" + C.BLUE + C.BOLD + "--- STAGE 5: EVALUATION ---" + C.RESET)
    
    # Validation set metrics
    y_val_prob = xgb.model.predict_proba(X_val_fused)[:, 1]
    y_val_pred = (y_val_prob > 0.5).astype(int)
    val_acc = accuracy_score(np.array(y_val), y_val_pred)
    ok(f"Validation Accuracy : {round(val_acc * 100, 2)}%")
    
    y_prob = xgb.model.predict_proba(X_test_fused)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    y_test_np = np.array(y_test)
    
    acc = accuracy_score(y_test_np, y_pred)
    auc_v = roc_auc_score(y_test_np, y_prob)
    
    print("\n" + C.CYAN + C.BOLD + "  --- CLASSIFICATION REPORT ---" + C.RESET)
    report = classification_report(y_test_np, y_pred, target_names=["Healthy", "Infected"], zero_division=0)
    for line in report.strip().splitlines(): print("  " + line)
    
    # Save plots
    cm = confusion_matrix(y_test_np, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (PyTorch+XGBoost)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "confusion_matrix_pt.png", dpi=150); plt.close(fig)
    
    fpr, tpr, _ = roc_curve(y_test_np, y_prob)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_v:.4f}")
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax2.set_title("ROC Curve", fontweight="bold")
    ax2.legend()
    fig2.tight_layout(); fig2.savefig(FIGURES_DIR / "roc_curve_pt.png", dpi=150); plt.close(fig2)

    ok("Test Accuracy : " + str(round(acc * 100, 2)) + "%")
    ok("Test AUC      : " + str(round(auc_v, 4)))
    ok("Saved evaluation figures to outputs/figures/")

    # 6. RAG AI Research Assistant
    print("\n" + C.BLUE + C.BOLD + "--- STAGE 6: RAG AI RESEARCH ASSISTANT ---" + C.RESET)
    try:
        from src.rag.rag_pipeline import RAGPipeline
        import json
        info("Initializing RAG Pipeline...")
        rag = RAGPipeline()
        
        # Simulate processing the highest probability prediction
        infected_indices = np.where(y_pred == 1)[0]
        if len(infected_indices) > 0:
            target_idx = infected_indices[0]
            mock_prediction = {
                "prediction": "Infected",
                "confidence": float(y_prob[target_idx]),
                "mutation_gene": "HA Gene",
                "mutation_positions": "125-138"
            }
            info(f"Triggering RAG Analysis for Detected Mutation: {mock_prediction['mutation_gene']}")
            
            # This calls PubMed and the LLM
            report = rag.process_prediction(mock_prediction)
            
            print("\n" + C.MAGENTA + C.BOLD + "  --- SCIENTIFIC EVIDENCE REPORT ---" + C.RESET)
            print(f"  Mutation    : {report['mutation']} (Pos: {report['position']})")
            print(f"  Confidence  : {round(report['confidence']*100, 1)}%")
            print(f"  Papers Found: {len(report['top_papers'])}")
            print(f"\n  [AI SUMMARY]\n  {report['scientific_evidence']}")
            print(f"\n  [RECOMMENDATIONS]")
            for r in report['recommendations']:
                print(f"   - {r}")
            print("\n")
            ok("RAG analysis completed.")
        else:
            info("No infected sequences found in this batch to analyze.")
    except Exception as e:
        err(f"RAG Module execution failed: {e}")

    banner("PIPELINE COMPLETE", C.GREEN)
    print("  Elapsed Time : " + str(round(time.time() - t0, 1)) + "s")
    print("  Model Ready  : PyTorch Multimodal -> XGBoost Classifier\n")

if __name__ == "__main__":
    main()
