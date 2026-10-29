"""
generate_paper_figures.py
=========================
Automated publication-quality figure generator for:
  "AvianNet: An Explainable Hybrid Deep Learning Fusion Framework for
   Detecting Avian Influenza from Raw Genomic Sequences"
  IEEE Conference Paper -- Section III: Proposed Methodology

Generates:
  figures/aviannet_architecture.png   -- AvianNet block-schematic
  figures/trifold_xai_framework.png   -- Tri-Fold XAI conceptual flow

Dependencies:
  pip install matplotlib numpy

Usage:
  python generate_paper_figures.py
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, FancyArrow
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTPUT_DIR = "figures"
DPI = 300
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# COLOUR PALETTE (colour-blind friendly, IEEE-compatible)
# -----------------------------------------------------------------
C_INPUT   = "#2C3E50"   # dark slate  - input / output nodes
C_BERT    = "#1A6B8A"   # teal        - Pathway 1 DNABERT-2
C_CNN     = "#2E7D32"   # forest grn  - Pathway 2 CNN+BiLSTM
C_GAT     = "#6A1B9A"   # deep purple - Pathway 3 GAT
C_FUSION  = "#B85C00"   # burnt org   - Projection MLP fusion
C_XGB     = "#B71C1C"   # dark red    - XGBoost meta-classifier
C_OUT     = "#37474F"   # charcoal    - prediction output
C_ARROW   = "#424242"
C_BG      = "#FAFAFA"
C_FOLD1   = "#1565C0"   # XAI fold 1
C_FOLD2   = "#00695C"   # XAI fold 2
C_FOLD3   = "#AD1457"   # XAI fold 3
WHITE     = "#FFFFFF"


# ===================================================================
# HELPER UTILITIES
# ===================================================================

def rounded_box(ax, x, y, w, h, color, label, sublabel=None,
                fontsize=8.5, text_color=WHITE, radius=0.04,
                alpha=1.0, bold=False):
    """Draw a rounded-rectangle block with label and optional sublabel."""
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=1.0, edgecolor=WHITE,
        facecolor=color, alpha=alpha, zorder=3
    )
    ax.add_patch(fancy)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4,
            wrap=False)
    if sublabel:
        ax.text(x, y - 0.045, sublabel,
                ha="center", va="center", fontsize=7.0,
                color=text_color, alpha=0.88, zorder=4, style="italic")


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.4,
          arrowstyle="-|>", mutation_scale=10):
    """Draw a fancy arrow between two points."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    color=color, lw=lw,
                    mutation_scale=mutation_scale,
                    connectionstyle="arc3,rad=0.0"
                ), zorder=5)


def dim_label(ax, x, y, txt, color="#555555"):
    ax.text(x, y, txt, ha="center", va="center",
            fontsize=6.5, color=color,
            fontstyle="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#CCCCCC", lw=0.5, alpha=0.85))


def section_bracket(ax, x, y_top, y_bot, label, color, lw=1.5):
    """Draw a vertical bracket on the left with a rotated label."""
    ax.plot([x, x], [y_bot, y_top], color=color, lw=lw,
            solid_capstyle="round", zorder=2)
    ax.plot([x, x + 0.018], [y_top, y_top], color=color, lw=lw, zorder=2)
    ax.plot([x, x + 0.018], [y_bot, y_bot], color=color, lw=lw, zorder=2)
    ax.text(x - 0.03, (y_top + y_bot) / 2, label,
            ha="center", va="center", fontsize=7, color=color,
            fontweight="bold", rotation=90, zorder=5)


# ===================================================================
# FIGURE 1 -- AvianNet Architecture
# ===================================================================

def make_architecture_figure():
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.965, "AvianNet End-to-End Architecture",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=C_INPUT)

    BW = 0.155   # block width
    BH = 0.085   # block height
    PBH = 0.075  # pathway block height

    Y_IN    = 0.86
    Y_P1    = 0.68
    Y_P2    = 0.50
    Y_P3    = 0.32
    Y_FUSE  = 0.50
    Y_XGB   = 0.50
    Y_OUT   = 0.50

    X_IN    = 0.12
    X_BPE   = 0.28
    X_ENC   = 0.455
    X_PROJ  = 0.645
    X_XGB   = 0.800
    X_OUT   = 0.940

    X_P1_TOK = 0.28
    X_P2_TOK = 0.28
    X_P3_TOK = 0.28

    # INPUT
    rounded_box(ax, X_IN, Y_IN, BW, BH, C_INPUT,
                "Raw Genomic DNA", "FASTA (unaligned)", bold=True)

    # Three-way split fan-out arrows
    for yt in [Y_P1, Y_P2, Y_P3]:
        ax.annotate("", xy=(X_BPE - BW / 2 - 0.005, yt),
                    xytext=(X_IN + BW / 2, Y_IN),
                    arrowprops=dict(
                        arrowstyle="-|>", color=C_ARROW,
                        lw=1.3, mutation_scale=9,
                        connectionstyle="arc3,rad=0.0"
                    ), zorder=5)

    # PATHWAY 1 - DNABERT-2
    rounded_box(ax, X_P1_TOK, Y_P1, BW, PBH, C_BERT,
                "BPE Tokeniser", "k-mer hash fallback")
    arrow(ax, X_P1_TOK + BW / 2, Y_P1, X_ENC - BW / 2, Y_P1, C_BERT)
    rounded_box(ax, X_ENC, Y_P1, BW, PBH, C_BERT,
                "DNABERT-2", "768-dim CLS embed", bold=True)
    dim_label(ax, X_ENC, Y_P1 - PBH / 2 - 0.045,
              r"$H_{\mathrm{BERT}}\in\mathbb{R}^{B\times768}$", C_BERT)

    # PATHWAY 2 - 1D-CNN+BiLSTM
    rounded_box(ax, X_P2_TOK, Y_P2, BW, PBH, C_CNN,
                "1-hot Encoding", "stride-1, len-pad")
    arrow(ax, X_P2_TOK + BW / 2, Y_P2, X_ENC - BW / 2, Y_P2, C_CNN)
    rounded_box(ax, X_ENC, Y_P2, BW, PBH, C_CNN,
                "1D-CNN + BiLSTM", "64-dim pool output", bold=True)
    dim_label(ax, X_ENC, Y_P2 - PBH / 2 - 0.045,
              r"$H_{\mathrm{CNN}}\in\mathbb{R}^{B\times64}$", C_CNN)

    # PATHWAY 3 - GAT
    rounded_box(ax, X_P3_TOK, Y_P3, BW, PBH, C_GAT,
                "Seq->Graph Build", "10 nodes, bi-edges")
    arrow(ax, X_P3_TOK + BW / 2, Y_P3, X_ENC - BW / 2, Y_P3, C_GAT)
    rounded_box(ax, X_ENC, Y_P3, BW, PBH, C_GAT,
                "GAT (2L, 4-head)", "128-dim mean-pool", bold=True)
    dim_label(ax, X_ENC, Y_P3 - PBH / 2 - 0.045,
              r"$H_{\mathrm{GAT}}\in\mathbb{R}^{B\times128}$", C_GAT)

    # Convergence arrows to CONCAT
    for yt in [Y_P1, Y_P2, Y_P3]:
        ax.annotate("", xy=(X_PROJ - BW / 2 - 0.005, Y_FUSE),
                    xytext=(X_ENC + BW / 2, yt),
                    arrowprops=dict(
                        arrowstyle="-|>", color=C_ARROW,
                        lw=1.2, mutation_scale=8,
                        connectionstyle="arc3,rad=0.0"
                    ), zorder=5)

    # Concat label
    ax.text(X_PROJ - BW / 2 - 0.022, Y_FUSE, "||",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=C_FUSION, zorder=6)

    # Projection MLP
    rounded_box(ax, X_PROJ, Y_FUSE, BW + 0.01, BH, C_FUSION,
                "Projection MLP", "960->512->256->128", bold=True)
    dim_label(ax, X_PROJ, Y_FUSE - BH / 2 - 0.045,
              r"$Z\in\mathbb{R}^{B\times128}$   BN-ReLU-Dropout",
              C_FUSION)

    arrow(ax, X_PROJ + (BW + 0.01) / 2, Y_FUSE,
          X_XGB - BW / 2, Y_XGB, C_FUSION, lw=1.5)

    # XGBoost
    rounded_box(ax, X_XGB, Y_XGB, BW, BH, C_XGB,
                "XGBoost", "n=200, d=6, lr=0.05", bold=True)
    dim_label(ax, X_XGB, Y_XGB - BH / 2 - 0.045,
              "scale_pos_weight~0.36", C_XGB)
    arrow(ax, X_XGB + BW / 2, Y_XGB, X_OUT - 0.05, Y_OUT, C_XGB, lw=1.5)

    # Output
    rounded_box(ax, X_OUT, Y_OUT, 0.095, BH, C_OUT,
                "Prediction", "H5N1 / Non-H5N1", bold=True)

    # Pathway bracket labels
    section_bracket(ax, 0.025, Y_P1 + PBH / 2 + 0.02,
                    Y_P3 - PBH / 2 - 0.02,
                    "Representation Pathways", C_INPUT)

    # Performance badge
    badge = FancyBboxPatch((0.62, 0.075), 0.36, 0.11,
                           boxstyle="round,pad=0.01,rounding_size=0.02",
                           linewidth=1, edgecolor="#888888",
                           facecolor="#F5F5F5", zorder=3)
    ax.add_patch(badge)
    ax.text(0.800, 0.155, "Verified Test Performance",
            ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="#333333", zorder=4)
    ax.text(0.800, 0.117,
            "Accuracy 97.37%  .  ROC-AUC 99.01%  .  MCC 0.9329",
            ha="center", va="center", fontsize=7,
            color="#555555", zorder=4)

    # Legend
    legend_items = [
        mpatches.Patch(color=C_BERT,   label="Pathway 1: DNABERT-2"),
        mpatches.Patch(color=C_CNN,    label="Pathway 2: 1D-CNN+BiLSTM"),
        mpatches.Patch(color=C_GAT,    label="Pathway 3: GAT"),
        mpatches.Patch(color=C_FUSION, label="Projection MLP"),
        mpatches.Patch(color=C_XGB,    label="XGBoost Classifier"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC",
              ncol=2, handlelength=1.2, columnspacing=0.8,
              bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(OUTPUT_DIR, "aviannet_architecture.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  [OK] Saved: " + out_path)
    return out_path


# ===================================================================
# FIGURE 2 -- Tri-Fold XAI Framework
# ===================================================================

def make_xai_figure():
    fig = plt.figure(figsize=(8.5, 5.4))
    fig.patch.set_facecolor(C_BG)

    gs = GridSpec(1, 4, figure=fig,
                  left=0.04, right=0.98, top=0.88, bottom=0.04,
                  wspace=0.06)

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis("off")
    ax_main.set_facecolor(C_BG)

    fig.text(0.5, 0.95, "Tri-Fold Explainability (XAI) Framework",
             ha="center", va="center", fontsize=11,
             fontweight="bold", color=C_INPUT)
    fig.text(0.5, 0.905,
             "Attention Discovery  ->  TreeSHAP Attribution  ->  Perturbation Fidelity Verification",
             ha="center", va="center", fontsize=8.5,
             color="#555555", fontstyle="italic")

    TOP_Y  = 0.82
    MID_Y  = 0.50
    BOT_Y  = 0.18
    BW     = 0.20
    BH     = 0.12
    ABH    = 0.085

    # Input node
    rounded_box(ax_main, 0.08, MID_Y, 0.12, BH + 0.02,
                C_INPUT, "Trained\nAvianNet\nModel", bold=True, fontsize=8)

    # Arrow to Fold 1
    arrow(ax_main, 0.14, MID_Y, 0.255, MID_Y, C_ARROW, lw=1.6)

    # FOLD 1
    FX1 = 0.34
    rounded_box(ax_main, FX1, MID_Y, BW + 0.02, BH + 0.04,
                C_FOLD1, "FOLD 1", bold=True, fontsize=9.5)

    rounded_box(ax_main, FX1, TOP_Y, BW + 0.04, ABH,
                C_FOLD1, "Multi-Head Attention", "Extraction", alpha=0.78)
    ax_main.annotate("", xy=(FX1, MID_Y + (BH + 0.04) / 2),
                     xytext=(FX1, TOP_Y - ABH / 2),
                     arrowprops=dict(arrowstyle="-|>", color=C_FOLD1,
                                     lw=1.0, mutation_scale=7))

    rounded_box(ax_main, FX1, BOT_Y, BW + 0.04, ABH,
                C_FOLD1, "Candidate Pathogenic", "Loci Ranking", alpha=0.78)
    ax_main.annotate("", xy=(FX1, BOT_Y + ABH / 2),
                     xytext=(FX1, MID_Y - (BH + 0.04) / 2),
                     arrowprops=dict(arrowstyle="-|>", color=C_FOLD1,
                                     lw=1.0, mutation_scale=7))

    ax_main.text(FX1, MID_Y - 0.002,
                 "High-weight\nsequence positions",
                 ha="center", va="center", fontsize=6.8,
                 color=WHITE, zorder=6)

    # Arrow Fold 1 -> Fold 2
    arrow(ax_main, FX1 + (BW + 0.02) / 2, MID_Y, 0.535, MID_Y,
          C_ARROW, lw=1.6)

    # FOLD 2
    FX2 = 0.615
    rounded_box(ax_main, FX2, MID_Y, BW + 0.02, BH + 0.04,
                C_FOLD2, "FOLD 2", bold=True, fontsize=9.5)

    rounded_box(ax_main, FX2, TOP_Y, BW + 0.04, ABH,
                C_FOLD2, "TreeSHAP Attribution", "on XGBoost Trees", alpha=0.78)
    ax_main.annotate("", xy=(FX2, MID_Y + (BH + 0.04) / 2),
                     xytext=(FX2, TOP_Y - ABH / 2),
                     arrowprops=dict(arrowstyle="-|>", color=C_FOLD2,
                                     lw=1.0, mutation_scale=7))

    rounded_box(ax_main, FX2, BOT_Y, BW + 0.04, ABH,
                C_FOLD2, "Shapley Value", r"$\phi_i$ per Latent Feature",
                alpha=0.78)
    ax_main.annotate("", xy=(FX2, BOT_Y + ABH / 2),
                     xytext=(FX2, MID_Y - (BH + 0.04) / 2),
                     arrowprops=dict(arrowstyle="-|>", color=C_FOLD2,
                                     lw=1.0, mutation_scale=7))

    ax_main.text(FX2, MID_Y - 0.002,
                 "128-dim latent\nfeature space",
                 ha="center", va="center", fontsize=6.8,
                 color=WHITE, zorder=6)

    # Arrow Fold 2 -> Fold 3
    arrow(ax_main, FX2 + (BW + 0.02) / 2, MID_Y, 0.815, MID_Y,
          C_ARROW, lw=1.6)

    # FOLD 3
    FX3 = 0.895
    rounded_box(ax_main, FX3, MID_Y, BW - 0.01, BH + 0.04,
                C_FOLD3, "FOLD 3", bold=True, fontsize=9.5)

    rounded_box(ax_main, FX3, TOP_Y, BW + 0.01, ABH,
                C_FOLD3, "In Silico Perturbation", "Fidelity Test", alpha=0.78)
    ax_main.annotate("", xy=(FX3, MID_Y + (BH + 0.04) / 2),
                     xytext=(FX3, TOP_Y - ABH / 2),
                     arrowprops=dict(arrowstyle="-|>", color=C_FOLD3,
                                     lw=1.0, mutation_scale=7))

    rounded_box(ax_main, FX3, BOT_Y, BW + 0.01, ABH,
                C_FOLD3, "Prediction Degradation",
                r"Cohen's $d$ Benchmark", alpha=0.78)
    ax_main.annotate("", xy=(FX3, BOT_Y + ABH / 2),
                     xytext=(FX3, MID_Y - (BH + 0.04) / 2),
                     arrowprops=dict(arrowstyle="-|>", color=C_FOLD3,
                                     lw=1.0, mutation_scale=7))

    ax_main.text(FX3, MID_Y - 0.002,
                 "Top-k SHAP\nfeature masking",
                 ha="center", va="center", fontsize=6.8,
                 color=WHITE, zorder=6)

    # Fold labels
    for fx, label, color in [
        (FX1, "Attention\nLoci Discovery", C_FOLD1),
        (FX2, "Quantitative\nAttribution", C_FOLD2),
        (FX3, "Causal\nVerification", C_FOLD3),
    ]:
        ax_main.text(fx, TOP_Y + ABH / 2 + 0.07, label,
                     ha="center", va="bottom", fontsize=7.5,
                     color=color, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec=color, lw=0.8, alpha=0.7))

    # Bottom summary band
    band = FancyBboxPatch((0.145, 0.01), 0.845, 0.095,
                          boxstyle="round,pad=0.01,rounding_size=0.015",
                          linewidth=0.8, edgecolor="#AAAAAA",
                          facecolor="#EEEEEE", zorder=2)
    ax_main.add_patch(band)
    ax_main.text(0.568, 0.058,
                 "Verification Criterion: SHAP-ranked features must cause statistically "
                 "greater prediction degradation than random masking\n"
                 "(Cohen's d benchmark; top-k = 10 features; pilot N = 15 sequences)",
                 ha="center", va="center", fontsize=7,
                 color="#333333", zorder=5)

    out_path = os.path.join(OUTPUT_DIR, "trifold_xai_framework.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  [OK] Saved: " + out_path)
    return out_path


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  AvianNet -- Publication Figure Generator")
    print("=" * 58)
    print("  Output directory : " + os.path.abspath(OUTPUT_DIR))
    print("  Resolution       : " + str(DPI) + " DPI")
    print("-" * 58)

    p1 = make_architecture_figure()
    p2 = make_xai_figure()

    print("-" * 58)
    print("  All figures generated successfully.")
    print("  Fig 1 -> " + str(p1))
    print("  Fig 2 -> " + str(p2))
    print("=" * 58)
