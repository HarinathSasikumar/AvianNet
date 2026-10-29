"""
SHAP Analysis — Avian DNA Classifier
======================================
Generates three publication-quality charts:
  1. SHAP Summary Plot (beeswarm)
  2. SHAP Global Feature Importance Bar Chart
  3. SHAP Waterfall Plot (for a single Infected prediction)

All charts saved as PNG (150 dpi) + PDF in outputs/research/xai/.

STRICTLY ADDITIVE — modifies no existing module.
Uses the pre-processed k-mer feature matrices from data/processed/.
A lightweight XGBoost is trained *solely* for XAI; it does not
touch or replace the production PyTorch / XGBoost fusion pipeline.
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
XAI_OUT      = PROJECT_ROOT / "outputs" / "research" / "xai"
XAI_OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

# ── Dark theme palette ─────────────────────────────────────────────
DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
TEXT_CLR  = "#e2e8f0"
MUTED_CLR = "#94a3b8"
BORDER    = "#334155"
CYAN      = "#06b6d4"
GREEN     = "#10b981"
RED       = "#f87171"


def _feature_names(n: int) -> list[str]:
    """Build readable 3-mer position labels for n features."""
    bases  = ["A", "T", "G", "C"]
    kmers  = [a + b + c for a in bases for b in bases for c in bases]
    labels = []
    for i in range(n):
        labels.append(f"{kmers[i % len(kmers)]}_p{i // len(kmers)}")
    return labels


def _train_xai_model(X_train, y_train, sample_size=600):
    """
    Train a lightweight XGBoost exclusively for SHAP explanations.
    This is an XAI helper model — separate from the production pipeline.
    """
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_class_weight

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), min(sample_size, len(X_train)), replace=False)
    Xs, ys = X_train[idx].astype(float), y_train[idx]

    classes = np.unique(ys)
    weights = compute_class_weight("balanced", classes=classes, y=ys)
    spw = float(weights[1] / weights[0]) if len(weights) > 1 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42,
        # Explicitly set base_score as float to fix SHAP compatibility
        # with newer XGBoost versions that store it as '[5E-1]'
        base_score=0.5,
        eval_metric="auc", verbosity=0,
    )
    model.fit(Xs, ys)
    return model


def _make_explainer(model, X_train, y_train):
    """
    Build a SHAP TreeExplainer.
    Strategy (in order of preference):
      1. shap.TreeExplainer(model.get_booster()) -- bypasses base_score string bug
      2. shap.TreeExplainer(model)               -- standard path
      3. Train a RandomForestClassifier and use TreeExplainer on that
         (always works, never falls back to slow PermutationExplainer)
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier

    # Strategy 1: use raw XGBoost booster (avoids base_score '[5E-1]' bug)
    try:
        booster = model.get_booster()
        exp = shap.TreeExplainer(booster)
        _ = exp.shap_values(X_train[:2].astype(float))
        return exp, "xgb_booster"
    except Exception:
        pass

    # Strategy 2: standard XGBClassifier path
    try:
        exp = shap.TreeExplainer(model)
        _ = exp.shap_values(X_train[:2].astype(float))
        return exp, "xgb_wrapper"
    except Exception:
        pass

    # Strategy 3: fast RandomForest (never fails, never uses PermutationExplainer)
    print("  [SHAP] Falling back to RandomForest explainer...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), min(300, len(X_train)), replace=False)
    rf  = RandomForestClassifier(n_estimators=50, max_depth=6,
                                  random_state=42, n_jobs=-1)
    rf.fit(X_train[idx].astype(float), y_train[idx])
    exp = shap.TreeExplainer(rf)
    return exp, "random_forest"


def _apply_dark_axes(ax, title: str, xlabel: str = ""):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED_CLR, labelpad=6)
    ax.tick_params(colors=MUTED_CLR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_shap_analysis() -> dict:
    """
    Full SHAP analysis pipeline.
    Returns dict with paths to generated PNG/PDF and top-feature data.
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap package not installed. Run: pip install shap"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("[XAI/SHAP] Loading processed feature matrices...")
    X_train = np.load(DATA_DIR / "train" / "X_train.npy").astype(float)
    y_train = np.load(DATA_DIR / "train" / "y_train.npy")
    X_test  = np.load(DATA_DIR / "test"  / "X_test.npy").astype(float)
    y_test  = np.load(DATA_DIR / "test"  / "y_test.npy")

    feat_names = _feature_names(X_train.shape[1])

    # Sub-sample test set for SHAP speed (SHAP is O(n²) with many features)
    rng = np.random.default_rng(42)
    shap_n   = min(200, len(X_test))
    shap_idx = rng.choice(len(X_test), shap_n, replace=False)
    X_shap   = X_test[shap_idx]
    y_shap   = y_test[shap_idx]

    print("[XAI/SHAP] Training XAI helper model...")
    model = _train_xai_model(X_train, y_train)

    print("[XAI/SHAP] Building SHAP explainer (booster -> wrapper -> RF fallback)...")
    explainer, exp_type = _make_explainer(model, X_train, y_train)

    print(f"[XAI/SHAP] Computing SHAP values (mode={exp_type})...")
    raw = explainer.shap_values(X_shap)

    # ── Robust sv normalisation ───────────────────────────────────
    # Handles: list-of-arrays (TreeExplainer), shap.Explanation (generic),
    # raw 3-D ndarray (PermutationExplainer), and plain 2-D ndarray.
    def _to_2d(raw_val):
        """Always return a (samples, n_features) 2-D float array for Infected class."""
        if isinstance(raw_val, list):
            arr = np.asarray(raw_val[1] if len(raw_val) == 2 else raw_val[0])
        elif hasattr(raw_val, "values"):         # shap.Explanation object
            arr = np.asarray(raw_val.values)
        else:
            arr = np.asarray(raw_val)

        if arr.ndim == 3:                        # (samples, features, classes)
            return arr[:, :, 1].astype(float)    # take Infected class only
        if arr.ndim == 2:
            return arr.astype(float)
        # Unexpected shape — flatten everything after axis 0
        return arr.reshape(arr.shape[0], -1).astype(float)

    sv_2d = _to_2d(raw)                          # guaranteed (samples, n_features)
    sv    = sv_2d                                # keep sv alias for summary_plot

    results: dict = {}

    # ── 1. Summary Plot (beeswarm) ─────────────────────────────────
    print("[XAI/SHAP] 1/3 Summary Plot...")
    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(figsize=(11, 7))
    fig1.patch.set_facecolor(DARK_BG)
    shap.summary_plot(
        sv, X_shap,
        feature_names=feat_names,
        max_display=20,
        show=False,
        plot_type="dot",
        color_bar=True,
    )
    fig1_new = plt.gcf()
    fig1_new.patch.set_facecolor(DARK_BG)
    fig1_new.axes[0].set_facecolor(CARD_BG)
    fig1_new.axes[0].set_title(
        "SHAP Summary — Top 20 k-mer Features",
        color=TEXT_CLR, fontsize=12, fontweight="bold", pad=10
    )
    fig1_new.axes[0].tick_params(colors=MUTED_CLR, labelsize=8)
    fig1_new.tight_layout()
    p1 = XAI_OUT / "shap_summary.png"
    p1p = XAI_OUT / "shap_summary.pdf"
    fig1_new.savefig(p1,  dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    fig1_new.savefig(p1p, bbox_inches="tight",          facecolor=DARK_BG)
    plt.close("all")
    results["shap_summary_png"] = str(p1)
    results["shap_summary_pdf"] = str(p1p)
    print(f"  [OK] {p1.name}")

    # ── 2. Feature Importance Bar Chart ───────────────────────────
    print("[XAI/SHAP] 2/3 Feature Importance Bar...")
    n_feats  = sv_2d.shape[1]                              # real feature count
    mean_abs = np.abs(sv_2d).mean(axis=0).ravel()          # shape (n_feats,)
    top_k    = min(20, n_feats)
    top_idx  = np.argsort(mean_abs)[-top_k:].tolist()       # plain Python ints, max = n_feats-1
    top_nm   = [feat_names[min(i, len(feat_names)-1)] for i in top_idx]
    top_vl   = [float(mean_abs[i]) for i in top_idx]

    fig2, ax2 = plt.subplots(figsize=(11, 7))
    fig2.patch.set_facecolor(DARK_BG)
    palette = plt.cm.cool(np.linspace(0.15, 0.95, top_k))
    ax2.barh(top_nm, top_vl, color=palette, height=0.65, edgecolor="none")
    _apply_dark_axes(ax2, "SHAP Global Feature Importance (Top 20 k-mers)",
                     "Mean |SHAP Value|")
    fig2.tight_layout()
    p2  = XAI_OUT / "shap_importance.png"
    p2p = XAI_OUT / "shap_importance.pdf"
    fig2.savefig(p2,  dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    fig2.savefig(p2p, bbox_inches="tight",          facecolor=DARK_BG)
    plt.close("all")
    results["shap_importance_png"] = str(p2)
    results["shap_importance_pdf"] = str(p2p)
    results["global_importance"]   = [
        {"feature": top_nm[-(i+1)], "value": float(top_vl[-(i+1)])}
        for i in range(min(10, top_k))
    ]
    print(f"  [OK] {p2.name}")

    # ── 3. Waterfall Plot (single Infected sample) ────────────────
    print("[XAI/SHAP] 3/3 Waterfall Plot...")
    infected_idx = np.where(y_shap == 1)[0]
    s_idx = int(infected_idx[0]) if len(infected_idx) else 0

    base_val = 0.5  # safe default
    try:
        ev = getattr(explainer, "expected_value", None)
        if ev is not None:
            if isinstance(ev, (list, np.ndarray)):
                base_val = float(np.array(ev).flat[0])
            else:
                base_val = float(ev)
    except Exception:
        pass

    wf_sv    = sv_2d[s_idx].ravel()                         # guaranteed 1-D
    top_w    = np.argsort(np.abs(wf_sv))[-15:].tolist()     # plain Python ints
    wf_names = [feat_names[i] for i in reversed(top_w)]
    wf_vals  = [float(wf_sv[i]) for i in reversed(top_w)]
    wf_clrs  = [GREEN if v > 0 else RED for v in wf_vals]

    fig3, ax3 = plt.subplots(figsize=(11, 7))
    fig3.patch.set_facecolor(DARK_BG)
    ax3.barh(wf_names, wf_vals, color=wf_clrs, height=0.6, edgecolor="none")
    ax3.axvline(0, color=BORDER, linewidth=1.2)
    ax3.text(0.01, 0.01,
             f"Base value: {base_val:.3f}",
             transform=ax3.transAxes,
             color=MUTED_CLR, fontsize=8)
    _apply_dark_axes(ax3,
                     f"SHAP Waterfall — Infected Sample #{s_idx}",
                     "SHAP Value  (impact on model output)")
    fig3.tight_layout()
    p3  = XAI_OUT / "shap_waterfall.png"
    p3p = XAI_OUT / "shap_waterfall.pdf"
    fig3.savefig(p3,  dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    fig3.savefig(p3p, bbox_inches="tight",          facecolor=DARK_BG)
    plt.close("all")
    results["shap_waterfall_png"] = str(p3)
    results["shap_waterfall_pdf"] = str(p3p)
    results["shap_base_value"]    = base_val
    results["shap_sample_idx"]    = s_idx
    print(f"  [OK] {p3.name}")

    # Save JSON manifest
    with open(XAI_OUT / "shap_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[XAI/SHAP] SHAP analysis complete.")
    return results
