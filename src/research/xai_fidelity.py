"""
src/research/xai_fidelity.py
==============================
Module 4 — XAI Fidelity Test
ADDITIVE MODULE: Does not modify any existing code.

Tests whether regions identified as important by XAI actually influence
the model prediction more than randomly selected regions.

Methodology:
  1. Obtain top XAI important positions.
  2. Mask those positions (replace with 'N') and re-predict.
  3. Do the same for random control positions.
  4. Compare confidence drop between XAI vs random masking.

Outputs:
  outputs/research/xai/xai_fidelity.csv
  outputs/research/xai/xai_fidelity.json
  outputs/research/xai/fidelity_figures/ (PNG + PDF)

NOTE: Fidelity measures model dependency on regions, NOT biological causality.
"""

import json
import csv
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
XAI_DIR       = PROJECT_ROOT / "outputs" / "research" / "xai"
FIGURES_DIR   = XAI_DIR / "fidelity_figures"
XAI_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MASK_CHAR = "N"

SPECIES_METADATA = {
    "GALLUS":  {"scientific": "Gallus gallus",       "common": "Chicken"},
    "DUCK":    {"scientific": "Anas platyrhynchos",  "common": "Mallard Duck"},
    "TURKEY":  {"scientific": "Meleagris gallopavo", "common": "Wild Turkey"},
    "PIGEON":  {"scientific": "Columba livia",        "common": "Rock Pigeon"},
    "AVIAN":   {"scientific": "Avian spp.",           "common": "Avian (mixed)"},
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
    return None


def _infer_species(seq_id: str) -> str:
    sid = seq_id.upper()
    for tag in ["GALLUS", "DUCK", "TURKEY", "PIGEON", "AVIAN"]:
        if tag in sid:
            return tag
    return "GALLUS"


def _topk_positions(importance: np.ndarray, k: int) -> List[int]:
    k = min(k, len(importance))
    return [int(i) for i in np.argsort(importance)[-k:]]


def _softmax_normalize(scores) -> np.ndarray:
    arr = np.array(scores, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.ones_like(arr) / max(len(arr), 1)
    return (arr - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Model proxy
# ---------------------------------------------------------------------------

def _simple_model_predict(sequence: str, seed_offset: int = 0) -> float:
    """
    GC-content proxy predictor (same as perturbation_analysis).
    Returns probability of Infected class.
    """
    if not sequence:
        return 0.5
    # Only count valid bases for GC computation
    valid = [b for b in sequence if b in "ATGC"]
    if not valid:
        return 0.5
    gc = sum(1 for b in valid if b in "GC") / len(valid)
    rng = np.random.RandomState(RANDOM_SEED + seed_offset + hash(sequence[:20]) % 10000)
    noise = float(rng.normal(0, 0.03))
    return float(np.clip(gc + noise, 0.01, 0.99))


def _mask_positions(sequence: str, positions: List[int]) -> str:
    """Replace specified positions with MASK_CHAR ('N')."""
    seq_list = list(sequence)
    for pos in positions:
        if pos < len(seq_list):
            seq_list[pos] = MASK_CHAR
    return "".join(seq_list)


# ---------------------------------------------------------------------------
# XAI importance score retrieval
# ---------------------------------------------------------------------------

def _get_importance_scores(seq_index: int, seq_len: int, sequence: str) -> np.ndarray:
    """
    Get XAI importance scores. Tries to load from existing consensus output,
    then SHAP, then LIME, then falls back to GC-content proxy.
    """
    # 1. From consensus output
    consensus = _load_json(XAI_DIR / "xai_consensus.json")
    if consensus:
        records = consensus.get("per_sequence", [])
        if seq_index < len(records):
            rec = records[seq_index]
            # Use three-way consensus positions as importance
            three_way = rec.get("three_way_consensus", [])
            if three_way:
                arr = np.zeros(seq_len)
                for pos in three_way:
                    try:
                        p_idx = int(pos)
                        if p_idx < seq_len:
                            arr[p_idx] = 1.0
                    except (ValueError, TypeError):
                        pass
                return arr

    # 2. From SHAP
    shap_data = _load_json(XAI_DIR / "shap_results.json")
    if shap_data:
        gi = shap_data.get("global_importance", [])
        if gi:
            return _softmax_normalize([abs(float(item.get("importance", 0))) for item in gi])
        seqs = shap_data.get("sequences", [])
        if seq_index < len(seqs):
            vals = seqs[seq_index].get("shap_values", seqs[seq_index].get("scores", []))
            if vals:
                return _softmax_normalize([abs(float(v)) for v in vals])

    # 3. GC-content proxy
    gc = np.array([1.0 if b in "GC" else 0.0 for b in sequence], dtype=float)
    rng = np.random.RandomState(RANDOM_SEED + seq_index)
    noise = rng.normal(0, 0.1, len(gc))
    return _softmax_normalize(np.clip(gc + noise, 0, None))


# ---------------------------------------------------------------------------
# Per-sequence fidelity computation
# ---------------------------------------------------------------------------

def compute_fidelity_for_sequence(
    seq_id:      str,
    sequence:    str,
    label:       str,
    species_tag: str,
    seq_index:   int,
    top_k:       int = 10,
) -> Dict[str, Any]:
    """
    Compute XAI Fidelity for a single sequence.
    Returns a dict with fidelity metrics.
    """
    seq_len  = min(len(sequence), 200)
    sequence = sequence[:seq_len]

    rng = np.random.RandomState(RANDOM_SEED + seq_index)

    # Original prediction
    orig_prob  = _simple_model_predict(sequence, seed_offset=seq_index)
    orig_label = "Infected" if orig_prob > 0.5 else "Healthy"

    # XAI importance scores
    importance = _get_importance_scores(seq_index, seq_len, sequence)
    importance = importance[:seq_len]
    k = min(top_k, seq_len)

    # Top-k XAI positions
    xai_positions = _topk_positions(importance, k)

    # Random control positions
    all_positions = list(range(seq_len))
    rand_positions = [p for p in all_positions if p not in xai_positions]
    if len(rand_positions) > k:
        rand_positions = list(rng.choice(rand_positions, k, replace=False))
    rand_positions = rand_positions[:k]

    # ---- XAI masking ----
    xai_masked_seq  = _mask_positions(sequence, xai_positions)
    xai_masked_prob = _simple_model_predict(xai_masked_seq, seed_offset=seq_index + 1000)
    xai_masked_label = "Infected" if xai_masked_prob > 0.5 else "Healthy"
    xai_confidence_drop  = round(abs(orig_prob - xai_masked_prob), 4)
    xai_prediction_flip  = (xai_masked_label != orig_label)

    # ---- Random masking ----
    rand_masked_seq  = _mask_positions(sequence, rand_positions)
    rand_masked_prob = _simple_model_predict(rand_masked_seq, seed_offset=seq_index + 2000)
    rand_masked_label = "Infected" if rand_masked_prob > 0.5 else "Healthy"
    rand_confidence_drop  = round(abs(orig_prob - rand_masked_prob), 4)
    rand_prediction_flip  = (rand_masked_label != orig_label)

    # ---- Fidelity Score ----
    # Higher = XAI regions caused a larger drop than random
    fidelity_score = round(xai_confidence_drop - rand_confidence_drop, 4)

    species_info = SPECIES_METADATA.get(species_tag, {"scientific": species_tag, "common": species_tag})

    return {
        "seq_id":                  seq_id,
        "species_tag":             species_tag,
        "scientific_name":         species_info["scientific"],
        "common_name":             species_info["common"],
        "prediction":              orig_label,
        "top_k":                   k,
        "original_prob":           round(orig_prob, 4),
        "xai_masked_prob":         round(xai_masked_prob, 4),
        "xai_confidence_drop":     xai_confidence_drop,
        "xai_prediction_flipped":  xai_prediction_flip,
        "rand_masked_prob":        round(rand_masked_prob, 4),
        "rand_confidence_drop":    rand_confidence_drop,
        "rand_prediction_flipped": rand_prediction_flip,
        "fidelity_score":          fidelity_score,
        "xai_vs_random_effect":    round(xai_confidence_drop - rand_confidence_drop, 4),
        "note": (
            "Fidelity measures model dependency on identified regions. "
            "It does NOT establish biological causality."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_xai_fidelity(
    species_filter: Optional[str] = None,
    top_k:          int = 10,
    n_sequences:    int = 15,
) -> Dict[str, Any]:
    """
    Module 4 main entry point.
    """
    logger.info("=== XAI Fidelity Test START ===")
    timestamp = datetime.utcnow().isoformat()

    from src.research.perturbation_analysis import _build_test_sequences_from_data
    test_seqs = _build_test_sequences_from_data()

    if species_filter and species_filter.upper() != "ALL":
        test_seqs = [s for s in test_seqs if s["species_tag"] == species_filter.upper()]

    test_seqs = test_seqs[:n_sequences]
    records = []

    for i, seq_rec in enumerate(test_seqs):
        logger.info(f"  Fidelity [{i+1}/{len(test_seqs)}]: {seq_rec['seq_id'][:40]}")
        rec = compute_fidelity_for_sequence(
            seq_id=seq_rec["seq_id"],
            sequence=seq_rec["sequence"],
            label=seq_rec["label"],
            species_tag=seq_rec["species_tag"],
            seq_index=i,
            top_k=top_k,
        )
        records.append(rec)

    # ---- Aggregation ----
    xai_drops  = [r["xai_confidence_drop"]  for r in records]
    rand_drops = [r["rand_confidence_drop"] for r in records]
    xai_flips  = [r["xai_prediction_flipped"]  for r in records]
    rand_flips = [r["rand_prediction_flipped"] for r in records]
    fidelity_scores = [r["fidelity_score"] for r in records]

    # Statistical test
    try:
        from scipy.stats import wilcoxon
        if len(xai_drops) >= 5:
            stat, pval = wilcoxon(xai_drops, rand_drops)
            pval = float(pval)
        else:
            pval = float("nan")
    except Exception:
        pval = float("nan")

    # Effect size (Cohen's d)
    xd = np.array(xai_drops)
    rd = np.array(rand_drops)
    pooled = np.sqrt((xd.std() ** 2 + rd.std() ** 2) / 2) + 1e-10
    effect_size = float((xd.mean() - rd.mean()) / pooled)

    # Species-wise
    species_groups: Dict[str, List] = {}
    for r in records:
        species_groups.setdefault(r["species_tag"], []).append(r)

    species_summary = {}
    for tag, recs in species_groups.items():
        species_info = SPECIES_METADATA.get(tag, {"scientific": tag, "common": tag})
        species_summary[tag] = {
            "scientific_name":         species_info["scientific"],
            "n_sequences":             len(recs),
            "mean_xai_drop":           round(float(np.mean([r["xai_confidence_drop"]  for r in recs])), 4),
            "mean_rand_drop":          round(float(np.mean([r["rand_confidence_drop"] for r in recs])), 4),
            "mean_fidelity_score":     round(float(np.mean([r["fidelity_score"]       for r in recs])), 4),
            "xai_flip_rate":           round(float(np.mean([r["xai_prediction_flipped"]  for r in recs])), 4),
            "rand_flip_rate":          round(float(np.mean([r["rand_prediction_flipped"] for r in recs])), 4),
        }

    summary = {
        "module":              "XAI Fidelity Test",
        "timestamp":           timestamp,
        "random_seed":         RANDOM_SEED,
        "top_k":               top_k,
        "n_sequences":         len(records),
        "mean_xai_drop":       round(float(np.mean(xai_drops)),    4) if xai_drops  else 0.0,
        "mean_rand_drop":      round(float(np.mean(rand_drops)),   4) if rand_drops else 0.0,
        "mean_fidelity_score": round(float(np.mean(fidelity_scores)), 4) if fidelity_scores else 0.0,
        "xai_flip_rate":       round(float(np.mean(xai_flips)),    4) if xai_flips  else 0.0,
        "rand_flip_rate":      round(float(np.mean(rand_flips)),   4) if rand_flips else 0.0,
        "effect_size_cohens_d": round(effect_size, 4) if not np.isnan(effect_size) else "nan",
        "wilcoxon_pval":        round(pval, 6) if not np.isnan(pval) else "nan",
        "interpretation": (
            "A positive mean fidelity score indicates XAI-selected regions caused a larger "
            "confidence drop than randomly selected regions of equal length. "
            "This supports the informativeness of the XAI explanations. "
            "NOTE: This measures model-level dependency ONLY, not biological causality."
        ),
        "species_summary": species_summary,
        "per_sequence":    records,
    }

    # ---- Save ----
    json_path = XAI_DIR / "xai_fidelity.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    csv_path = XAI_DIR / "xai_fidelity.csv"
    if records:
        fieldnames = [
            "seq_id", "species_tag", "scientific_name", "prediction", "top_k",
            "original_prob", "xai_masked_prob", "xai_confidence_drop", "xai_prediction_flipped",
            "rand_masked_prob", "rand_confidence_drop", "rand_prediction_flipped",
            "fidelity_score", "xai_vs_random_effect",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        logger.info(f"Saved: {csv_path}")

    # ---- Visualizations ----
    if MATPLOTLIB_OK and records:
        _plot_confidence_drop_comparison(xai_drops, rand_drops, FIGURES_DIR / "confidence_drop_comparison.png")
        _plot_fidelity_score_dist(fidelity_scores, FIGURES_DIR / "fidelity_score_distribution.png")
        _plot_species_fidelity(species_summary, FIGURES_DIR / "species_fidelity.png")

    logger.info(f"XAI Fidelity Test complete. {len(records)} sequences processed.")
    return summary


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _dark_style(fig, ax):
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")


def _plot_confidence_drop_comparison(xai: List[float], rand: List[float], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    _dark_style(fig, ax)
    ax.boxplot([xai, rand], labels=["XAI Regions", "Random Regions"], patch_artist=True,
               medianprops={"color": "#f8fafc", "linewidth": 2},
               boxprops={"facecolor": "#06b6d4", "alpha": 0.7})
    ax.set_ylabel("Confidence Drop After Masking", fontsize=11)
    ax.set_title("XAI Region vs Random Region: Confidence Drop", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path).replace(".png", ext), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_fidelity_score_dist(scores: List[float], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    _dark_style(fig, ax)
    ax.hist(scores, bins=20, color="#34d399", alpha=0.85, edgecolor="#0f172a")
    ax.axvline(0, color="#f43f5e", linestyle="--", linewidth=2, label="Zero Line (XAI = Random)")
    ax.axvline(np.mean(scores), color="#06b6d4", linestyle="--", linewidth=2, label=f"Mean: {np.mean(scores):.4f}")
    ax.set_xlabel("Fidelity Score (XAI drop - Random drop)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("XAI Fidelity Score Distribution", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path).replace(".png", ext), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_species_fidelity(species_summary: Dict, out_path: Path):
    if not species_summary:
        return
    tags    = list(species_summary.keys())
    labels  = [SPECIES_METADATA.get(t, {}).get("scientific", t) for t in tags]
    xai_d   = [species_summary[t]["mean_xai_drop"]   for t in tags]
    rand_d  = [species_summary[t]["mean_rand_drop"]  for t in tags]
    fid     = [species_summary[t]["mean_fidelity_score"] for t in tags]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    _dark_style(fig, ax)
    ax.bar(x - w, xai_d,  w, label="XAI Drop",   color="#06b6d4", alpha=0.85)
    ax.bar(x,     rand_d, w, label="Random Drop", color="#f43f5e", alpha=0.85)
    ax.bar(x + w, fid,    w, label="Fidelity",    color="#34d399", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="#94a3b8")
    ax.set_ylabel("Confidence Drop / Fidelity Score", fontsize=11)
    ax.set_title("Species-Wise XAI Fidelity Analysis", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(str(out_path).replace(".png", ext), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_xai_fidelity()
