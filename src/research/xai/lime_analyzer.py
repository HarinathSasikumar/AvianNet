"""
LIME Analysis — Avian DNA Classifier
======================================
Explains individual DNA sequence predictions using LIME
(Local Interpretable Model-agnostic Explanations).

Generates:
  1. LIME Explanation Bar Chart (positive / negative k-mer contributions)
  2. LIME Feature Contribution Table (top features, saved as PNG)

All saved as PNG (150 dpi) + PDF in outputs/research/xai/.

STRICTLY ADDITIVE — modifies no existing module.
Uses the same pre-processed k-mer feature matrices as the SHAP module.
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
XAI_OUT      = PROJECT_ROOT / "outputs" / "research" / "xai"
XAI_OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
TEXT_CLR  = "#e2e8f0"
MUTED_CLR = "#94a3b8"
BORDER    = "#334155"
GREEN     = "#10b981"
RED       = "#f87171"
CYAN      = "#06b6d4"


def _feature_names(n: int) -> list[str]:
    bases = ["A", "T", "G", "C"]
    kmers = [a + b + c for a in bases for b in bases for c in bases]
    return [f"{kmers[i % len(kmers)]}_p{i // len(kmers)}" for i in range(n)]


def _train_xai_model(X_train, y_train, sample_size=600):
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
        eval_metric="auc", verbosity=0, use_label_encoder=False
    )
    model.fit(Xs, ys)
    return model


def run_lime_analysis(n_samples: int = 3) -> dict:
    """
    Run LIME analysis on n_samples test instances.
    Generates a bar chart for each and a combined summary PNG.
    Returns dict with paths and feature contribution data.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return {"error": "lime package not installed. Run: pip install lime"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("[XAI/LIME] Loading processed feature matrices...")
    X_train = np.load(DATA_DIR / "train" / "X_train.npy").astype(float)
    y_train = np.load(DATA_DIR / "train" / "y_train.npy")
    X_test  = np.load(DATA_DIR / "test"  / "X_test.npy").astype(float)
    y_test  = np.load(DATA_DIR / "test"  / "y_test.npy")

    feat_names = _feature_names(X_train.shape[1])

    print("[XAI/LIME] Training XAI helper model...")
    model = _train_xai_model(X_train, y_train)

    print("[XAI/LIME] Building LIME Explainer...")
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feat_names,
        class_names=["Healthy", "Infected"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    # Pick one infected sample and one healthy sample to explain
    infected_indices = np.where(y_test == 1)[0]
    healthy_indices  = np.where(y_test == 0)[0]

    sample_indices = []
    if len(infected_indices): sample_indices.append(("Infected", int(infected_indices[0])))
    if len(healthy_indices):  sample_indices.append(("Healthy",  int(healthy_indices[0])))
    # Add a second infected for diversity
    if len(infected_indices) > 1: sample_indices.append(("Infected", int(infected_indices[5])))

    results: dict = {"explanations": [], "lime_chart_png": None, "lime_chart_pdf": None}
    top_n = 15  # features to display per sample

    # ── Combined multi-panel figure ───────────────────────────────
    n_panels = len(sample_indices)
    fig = plt.figure(figsize=(14, 6 * n_panels))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.55)

    for panel_i, (label, idx) in enumerate(sample_indices):
        instance = X_test[idx]
        true_cls  = int(y_test[idx])
        prob      = float(model.predict_proba([instance])[0, 1])

        print(f"  [LIME] Explaining {label} sample #{idx} (prob={prob:.3f})...")
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=top_n,
            num_samples=500,
        )

        # Extract contributions for the "Infected" class (index 1)
        contribs = exp.as_list(label=1)
        f_names  = [c[0] for c in contribs]
        f_vals   = [c[1] for c in contribs]
        bar_clrs = [GREEN if v > 0 else RED for v in f_vals]

        # Sort by absolute magnitude
        order    = np.argsort(np.abs(f_vals))
        f_names  = [f_names[i] for i in order]
        f_vals   = [f_vals[i]  for i in order]
        bar_clrs = [bar_clrs[i] for i in order]

        ax = fig.add_subplot(gs[panel_i])
        ax.set_facecolor(CARD_BG)
        ax.barh(f_names, f_vals, color=bar_clrs, height=0.65, edgecolor="none")
        ax.axvline(0, color=BORDER, linewidth=1.2)
        ax.set_title(
            f"LIME Explanation — Sample #{idx}  |  True: {label}  |  P(Infected)={prob:.1%}",
            color=TEXT_CLR, fontsize=11, fontweight="bold", pad=8
        )
        ax.set_xlabel("Feature Contribution to Infected Prediction", color=MUTED_CLR, labelpad=6)
        ax.tick_params(colors=MUTED_CLR, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_els = [Patch(color=GREEN, label="Supports Infected"),
                      Patch(color=RED,   label="Opposes Infected")]
        ax.legend(handles=legend_els, loc="lower right",
                  facecolor=CARD_BG, edgecolor=BORDER,
                  labelcolor=TEXT_CLR, fontsize=8)

        # Store for API / dashboard
        results["explanations"].append({
            "sample_idx":   idx,
            "true_label":   label,
            "prob_infected": prob,
            "top_features": [
                {"feature": f_names[-(i+1)], "contribution": f_vals[-(i+1)]}
                for i in range(min(10, len(f_names)))
            ],
        })

    fig.tight_layout()
    p1  = XAI_OUT / "lime_explanation.png"
    p1p = XAI_OUT / "lime_explanation.pdf"
    fig.savefig(p1,  dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    fig.savefig(p1p, bbox_inches="tight",          facecolor=DARK_BG)
    plt.close("all")
    results["lime_chart_png"] = str(p1)
    results["lime_chart_pdf"] = str(p1p)
    print(f"  [OK] {p1.name}")

    # ── Per-sample contribution table figure ─────────────────────
    for exp_data in results["explanations"]:
        _save_contribution_table(exp_data)

    # Save JSON manifest
    with open(XAI_OUT / "lime_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[XAI/LIME] LIME analysis complete.")
    return results


def _save_contribution_table(exp_data: dict):
    """Render the top-10 features as a nicely formatted table PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats  = exp_data["top_features"]
    rows   = [(f["feature"], f"{f['contribution']:+.4f}",
               "Positive" if f["contribution"] > 0 else "Negative")
              for f in feats[:10]]
    col_lbls = ["Feature (k-mer)", "Contribution", "Direction"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_lbls,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Style table cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(CARD_BG if row % 2 == 0 else "#172033")
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor("#1d3352")
            cell.set_text_props(color=CYAN, fontweight="bold")
        else:
            val = rows[row - 1][2] if row <= len(rows) else ""
            cell.set_text_props(color=GREEN if val == "Positive" else (RED if val == "Negative" else TEXT_CLR))

    ax.set_title(
        f"LIME Feature Contributions — Sample #{exp_data['sample_idx']} ({exp_data['true_label']})",
        color=TEXT_CLR, fontsize=10, fontweight="bold", pad=12
    )
    fig.tight_layout()
    fname = XAI_OUT / f"lime_table_{exp_data['sample_idx']}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close("all")
