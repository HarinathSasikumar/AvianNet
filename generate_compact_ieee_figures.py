"""
generate_compact_ieee_figures.py
================================
Publication-grade IEEE vector figure generator for:
"AvianNet: An Explainable Hybrid Deep Learning Fusion Framework for
 Detecting Avian Influenza from Raw Genomic Sequences"

Conforms strictly to IEEE Transactions & Conference standards:
- 85% monochrome / grayscale base with muted, desaturated functional accents.
- Tight bounding boxes (pad_inches=0.02, 300 DPI).
- Precise column dimensions:
  * Two-Column figures: 7.0 x 2.4 in / 7.0 x 2.2 in
  * Single-Column figures: 3.5 x 2.4 in / 3.5 x 2.0 in / 3.5 x 2.2 in
- Clean typography (Arial / DejaVu Sans, 7pt-9pt, no overlapping text).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# Ensure output directory exists
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Strict IEEE Typography and Layout Specs
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.family": "sans-serif",
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "text.color": "#1A202C",
    "axes.labelcolor": "#1A202C",
    "xtick.color": "#1A202C",
    "ytick.color": "#1A202C",
    "font.size": 7.5,
    "lines.linewidth": 0.8
})

# Refined IEEE Academic Palette (Desaturated, High-Contrast)
CHARCOAL = "#222222"
DARK_SLATE = "#2D3748"
MID_GRAY = "#718096"
LIGHT_GRAY = "#E2E8F0"
SURFACE_BG = "#F8FAFC"
BORDER_DARK = "#4A5568"
BORDER_LIGHT = "#CBD5E1"

# Functional Accents (Desaturated academic tones)
ACCENT_BLUE = "#2B6CB0"    # Pathway 1: Contextual
ACCENT_TEAL = "#2C7A7B"    # Pathway 2: Spatial/Sequential
ACCENT_PURPLE = "#4C51BF"  # Pathway 3: Topological Graph
ACCENT_CRIMSON = "#C53030" # Critical intervention / Perturbation drop
ACCENT_GREEN = "#2F855A"   # Statistical validation


def draw_ieee_card(ax, x, y, w, h, title=None, subtitle=None, bg=SURFACE_BG, border=BORDER_LIGHT,
                   title_col=CHARCOAL, sub_col=DARK_SLATE, lw=0.9, radius=0.015, title_sz=8.0, sub_sz=6.8, bold=True):
    """Draw a compact, high-precision IEEE rectangular box with crisp borders."""
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         facecolor=bg, edgecolor=border, linewidth=lw, zorder=3)
    ax.add_patch(box)
    
    if title and subtitle:
        w_title = "bold" if bold else "normal"
        ax.text(x + w / 2.0, y + h * 0.64, title, ha="center", va="center",
                fontsize=title_sz, fontweight=w_title, color=title_col, zorder=4)
        ax.text(x + w / 2.0, y + h * 0.32, subtitle, ha="center", va="center",
                fontsize=sub_sz, color=sub_col, zorder=4)
    elif title:
        w_title = "bold" if bold else "normal"
        ax.text(x + w / 2.0, y + h / 2.0, title, ha="center", va="center",
                fontsize=title_sz, fontweight=w_title, color=title_col, zorder=4)


def draw_ieee_arrow(ax, p1, p2, color=DARK_SLATE, lw=1.0):
    """Draw a thin, sharp IEEE directional connector."""
    arr = FancyArrowPatch(p1, p2, arrowstyle="->,head_width=2.5,head_length=3.5",
                          color=color, linewidth=lw, zorder=5)
    ax.add_patch(arr)


# ==============================================================================
# FIGURE 1: Core System Architecture (Two-Column: 7.0 x 2.4 in)
# ==============================================================================
def render_architecture_figure():
    fig, ax = plt.subplots(figsize=(7.0, 2.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header Banner (Clean, concise)
    ax.text(1, 95, "AvianNet Architecture", fontsize=9.0, fontweight="bold", color=CHARCOAL)
    ax.text(26, 95, "Raw Genomic Input -> Tri-Pathway Parallel Feature Extraction -> Projection MLP -> XGBoost Meta-Head",
            fontsize=7.2, color=MID_GRAY)

    # 1. Input Block
    draw_ieee_card(ax, 1, 14, 11, 74, "Input DNA", "Raw FASTA\nL=512 nt\n(Unaligned)",
                   bg="#F1F5F9", border=BORDER_DARK, lw=1.1, title_sz=8.0, sub_sz=6.8)

    # Pathway Enclosure
    lane_bg = FancyBboxPatch((13.5, 8), 44.5, 83, boxstyle="round,pad=0.0,rounding_size=0.015",
                            facecolor="#FCFDFE", edgecolor="#E2E8F0", linewidth=0.8, linestyle="--", zorder=1)
    ax.add_patch(lane_bg)

    # --- Pathway 1: Contextual Language Stream ---
    draw_ieee_card(ax, 15, 68, 17, 20, "Pathway 1: Contextual", "4-mer Hash Projection\nTokenizer Vocabulary",
                   bg="#F0F7FF", border=ACCENT_BLUE, title_col=ACCENT_BLUE, title_sz=7.5, sub_sz=6.5)
    draw_ieee_arrow(ax, (32, 78), (35, 78), color=ACCENT_BLUE)
    draw_ieee_card(ax, 35, 68, 21, 20, "Linear Embedding", r"$H_{BERT} \in \mathbb{R}^{B \times 768}$" + "\nGlobal Semantic Syntax",
                   bg="#FFFFFF", border=ACCENT_BLUE, title_col=ACCENT_BLUE, title_sz=7.5, sub_sz=6.5)

    # --- Pathway 2: Spatial-Sequential Stream ---
    draw_ieee_card(ax, 15, 39, 17, 21, "Pathway 2: Spatial CNN", "1-Hot Matrix (512x4)\nConv1D (768->128->64)",
                   bg="#F0FDF4", border=ACCENT_TEAL, title_col=ACCENT_TEAL, title_sz=7.5, sub_sz=6.5)
    draw_ieee_arrow(ax, (32, 49.5), (35, 49.5), color=ACCENT_TEAL)
    draw_ieee_card(ax, 35, 39, 21, 21, "MaxPool + BiLSTM", r"$H_{BiLSTM} \in \mathbb{R}^{B \times 64}$" + "\nLocalized Spatial Motifs",
                   bg="#FFFFFF", border=ACCENT_TEAL, title_col=ACCENT_TEAL, title_sz=7.5, sub_sz=6.5)

    # --- Pathway 3: Topological Graph Stream ---
    draw_ieee_card(ax, 15, 11, 17, 20, "Pathway 3: Graph Net", "10-Node Sequence Graph\nBidirectional Adjacency",
                   bg="#FAF5FF", border=ACCENT_PURPLE, title_col=ACCENT_PURPLE, title_sz=7.5, sub_sz=6.5)
    draw_ieee_arrow(ax, (32, 21), (35, 21), color=ACCENT_PURPLE)
    draw_ieee_card(ax, 35, 11, 21, 20, "4-Head GAT Layer", r"$H_{GAT} \in \mathbb{R}^{B \times 128}$" + "\nMean-Pooled Epistasis",
                   bg="#FFFFFF", border=ACCENT_PURPLE, title_col=ACCENT_PURPLE, title_sz=7.5, sub_sz=6.5)

    # Connect Input to 3 Pathways
    draw_ieee_arrow(ax, (12, 58), (15, 78), color=ACCENT_BLUE)
    draw_ieee_arrow(ax, (12, 51), (15, 49.5), color=ACCENT_TEAL)
    draw_ieee_arrow(ax, (12, 44), (15, 21), color=ACCENT_PURPLE)

    # --- Fusion Node (Concatenation) ---
    draw_ieee_card(ax, 59.5, 14, 9, 74, "Concat", r"$H_{concat}$" + "\n[768||64||128]\n\n" + r"$\mathbb{R}^{B \times 960}$",
                   bg="#FFFBEB", border="#B45309", title_col="#B45309", lw=1.0, title_sz=7.8, sub_sz=6.5)

    draw_ieee_arrow(ax, (56, 78), (59.5, 66), color=ACCENT_BLUE)
    draw_ieee_arrow(ax, (56, 49.5), (59.5, 51), color=ACCENT_TEAL)
    draw_ieee_arrow(ax, (56, 21), (59.5, 36), color=ACCENT_PURPLE)

    # --- Projection MLP ---
    draw_ieee_card(ax, 70, 14, 14.5, 74, "Projection MLP",
                   "Dense(960->512)\nBN+ReLU (p=0.3)\nDense(512->256)\nBN+ReLU (p=0.2)\nDense(256->128)\n\n" + r"$Z \in \mathbb{R}^{B \times 128}$",
                   bg="#F8FAFC", border=DARK_SLATE, lw=1.0, title_sz=7.8, sub_sz=6.5)
    draw_ieee_arrow(ax, (68.5, 51), (70, 51), color=DARK_SLATE)

    # --- XGBoost Classifier & Output ---
    draw_ieee_card(ax, 86, 20, 13, 62, "XGBoost Meta", "200 Trees\nMax Depth = 6\nscale_pos=0.36\nSubsample=0.8\n\nDecision:\nP(H5N1+)",
                   bg="#FFF5F5", border=ACCENT_CRIMSON, title_col=ACCENT_CRIMSON, lw=1.1, title_sz=8.0, sub_sz=6.5)
    draw_ieee_arrow(ax, (84.5, 51), (86, 51), color=DARK_SLATE)

    plt.tight_layout(pad=0.02)
    save_path = os.path.join(FIGURES_DIR, "aviannet_architecture.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


# ==============================================================================
# FIGURE 2: Tri-Fold Explainability Pipeline (Two-Column: 7.0 x 2.2 in)
# ==============================================================================
def render_trifold_xai_figure():
    fig, ax = plt.subplots(figsize=(7.0, 2.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header
    ax.text(1, 94, "Tri-Fold Explainability (XAI) Verification Protocol", fontsize=8.8, fontweight="bold", color=CHARCOAL)
    ax.text(48, 94, "Hypothesis Generation -> Quantitative Shapley Attribution -> Causal Perturbation Benchmark",
            fontsize=7.0, color=MID_GRAY)

    card_w = 30.5
    card_h = 82
    y_base = 7

    # --- FOLD 1: Hypothesis Discovery ---
    draw_ieee_card(ax, 1, y_base, card_w, card_h, bg="#FFFFFF", border=BORDER_DARK, lw=1.0)
    ax.text(1 + card_w / 2.0, y_base + card_h - 6, "Fold 1: Hypothesis Discovery", ha="center",
            fontsize=8.0, fontweight="bold", color=CHARCOAL)
    ax.text(1 + card_w / 2.0, y_base + card_h - 13, "Cross-Attention Matrix (512x512)", ha="center",
            fontsize=6.8, color=MID_GRAY)

    # Mini 5x5 Heatmap Representation
    mat_x = 4.0
    mat_y = 20
    mat_w = 11.5
    mat_h = 36
    np.random.seed(42)
    sample_attn = np.array([
        [0.05, 0.08, 0.12, 0.04, 0.06],
        [0.07, 0.45, 0.15, 0.08, 0.09],
        [0.10, 0.18, 0.65, 0.12, 0.08],
        [0.04, 0.09, 0.11, 0.05, 0.06],
        [0.08, 0.14, 0.22, 0.09, 0.11]
    ])
    cell_w = mat_w / 5.0
    cell_h = mat_h / 5.0
    for r in range(5):
        for c in range(5):
            val = sample_attn[r, c]
            gray_val = 1.0 - val * 0.95
            rect = Rectangle((mat_x + c * cell_w, mat_y + (4 - r) * cell_h), cell_w, cell_h,
                             facecolor=(gray_val, gray_val, gray_val), edgecolor="#CBD5E1", linewidth=0.5)
            ax.add_patch(rect)
    ax.text(mat_x + mat_w / 2.0, mat_y - 4, "Attention Matrix", ha="center", fontsize=6.2, color=DARK_SLATE)

    # Column Average Vector & Candidate Callout
    draw_ieee_arrow(ax, (mat_x + mat_w + 1.5, mat_y + mat_h / 2.0), (mat_x + mat_w + 3.5, mat_y + mat_h / 2.0), lw=0.9)
    col_x = mat_x + mat_w + 4.5
    col_w = 7.5
    draw_ieee_card(ax, col_x, mat_y + 4, col_w, 28, "Column Avg\nScore", r"$\bar{\alpha}_j \rightarrow$ Loci",
                   bg="#F1F5F9", border=BORDER_LIGHT, lw=0.7, title_sz=6.5, sub_sz=6.0)
    ax.text(1 + card_w / 2.0, y_base + 3, "Output: Ranked Candidate Loci", ha="center", fontsize=6.8, color=ACCENT_BLUE, fontweight="bold")

    # Arrow Fold 1 -> Fold 2
    draw_ieee_arrow(ax, (1 + card_w + 0.5, 48), (1 + card_w + 3.0, 48), color=CHARCOAL, lw=1.2)

    # --- FOLD 2: Quantitative Attribution ---
    f2_x = 35.5
    draw_ieee_card(ax, f2_x, y_base, card_w, card_h, bg="#FFFFFF", border=BORDER_DARK, lw=1.0)
    ax.text(f2_x + card_w / 2.0, y_base + card_h - 6, "Fold 2: Quantitative Attribution", ha="center",
            fontsize=8.0, fontweight="bold", color=CHARCOAL)
    ax.text(f2_x + card_w / 2.0, y_base + card_h - 13, "TreeSHAP on Latent Features Z (128-d)", ha="center",
            fontsize=6.8, color=MID_GRAY)

    # Mini Horizontal Bar Chart of Shapley Values
    bar_y_starts = [52, 44, 36, 28, 20]
    bar_widths = [16.5, 13.8, 11.2, 8.5, 6.0]
    bar_labels = [r"$z_{42}$", r"$z_{19}$", r"$z_{88}$", r"$z_{105}$", r"$z_{07}$"]
    for by, bw, bl in zip(bar_y_starts, bar_widths, bar_labels):
        ax.text(f2_x + 5.5, by + 1.5, bl, ha="right", va="center", fontsize=6.8, color=DARK_SLATE)
        r = Rectangle((f2_x + 6.5, by), bw, 4.5, facecolor="#4A5568", edgecolor=CHARCOAL, linewidth=0.6)
        ax.add_patch(r)
    ax.text(f2_x + 18, 14, r"Exact Shapley Value $|\phi_i|$", ha="center", fontsize=6.5, color=DARK_SLATE)
    ax.text(f2_x + card_w / 2.0, y_base + 3, "Output: Top-10 Salient Dimensions", ha="center", fontsize=6.8, color=ACCENT_BLUE, fontweight="bold")

    # Arrow Fold 2 -> Fold 3
    draw_ieee_arrow(ax, (f2_x + card_w + 0.5, 48), (f2_x + card_w + 3.0, 48), color=CHARCOAL, lw=1.2)

    # --- FOLD 3: Causal Verification ---
    f3_x = 69.5
    draw_ieee_card(ax, f3_x, y_base, card_w, card_h, bg="#FFFFFF", border=BORDER_DARK, lw=1.0)
    ax.text(f3_x + card_w / 2.0, y_base + card_h - 6, "Fold 3: Causal Verification", ha="center",
            fontsize=8.0, fontweight="bold", color=CHARCOAL)
    ax.text(f3_x + card_w / 2.0, y_base + card_h - 13, "In-Silico Feature Masking Intervention", ha="center",
            fontsize=6.8, color=MID_GRAY)

    # Comparison Delta Bars
    # Bar A: Targeted SHAP Drop
    ax.add_patch(Rectangle((f3_x + 6, 22), 6.5, 42, facecolor=ACCENT_CRIMSON, edgecolor=CHARCOAL, linewidth=0.7))
    ax.text(f3_x + 9.25, 66, r"$\Delta p=0.64$", ha="center", fontsize=6.5, fontweight="bold", color=ACCENT_CRIMSON)
    ax.text(f3_x + 9.25, 17, "Targeted", ha="center", fontsize=6.5, color=DARK_SLATE)

    # Bar B: Random Control Drop
    ax.add_patch(Rectangle((f3_x + 17, 22), 6.5, 6, facecolor="#CBD5E1", edgecolor=DARK_SLATE, linewidth=0.7))
    ax.text(f3_x + 20.25, 30, r"$\Delta p=0.05$", ha="center", fontsize=6.5, color=DARK_SLATE)
    ax.text(f3_x + 20.25, 17, "Random", ha="center", fontsize=6.5, color=DARK_SLATE)

    # Callout Badge: Cohen's d
    draw_ieee_card(ax, f3_x + 2.5, y_base + 3, card_w - 5, 9.5,
                   "Cohen's d = 1.42 (p < 0.001)", None,
                   bg="#F0FDF4", border=ACCENT_GREEN, title_col=ACCENT_GREEN, title_sz=7.0)

    plt.tight_layout(pad=0.02)
    save_path = os.path.join(FIGURES_DIR, "trifold_xai_framework.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


# ==============================================================================
# FIGURE 3: Pathway 3 Topological Graph Detail (Single-Column: 3.5 x 2.4 in)
# ==============================================================================
def render_gat_graph_figure():
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(2, 95, "Pathway 3: Sequence Graph Construction", fontsize=8.2, fontweight="bold", color=CHARCOAL)
    ax.text(2, 89, "Unaligned 512 nt -> 10 Sampled Nodes -> GAT Epistasis", fontsize=6.8, color=MID_GRAY)

    # Step 1: Sequence Bar with 10 Sampling Loci
    seq_y = 74
    ax.add_patch(Rectangle((4, seq_y), 92, 7, facecolor="#F1F5F9", edgecolor=BORDER_DARK, linewidth=0.8))
    ax.text(50, seq_y + 3.5, "Raw Sequence (L = 512 nucleotides)", ha="center", va="center", fontsize=6.8, color=DARK_SLATE)

    node_xs = np.linspace(8, 92, 10)
    for idx, nx in enumerate(node_xs):
        ax.plot([nx, nx], [seq_y, seq_y + 7], color="#94A3B8", lw=0.9)
        ax.text(nx, seq_y + 9.5, f"{idx+1}", ha="center", fontsize=5.8, color=DARK_SLATE)

    draw_ieee_arrow(ax, (50, 72), (50, 64), lw=0.8)

    # Step 2 & 3: 10 Nodes, Adjacency & Self-Loops
    circle_y = 50
    for idx, nx in enumerate(node_xs):
        # Self-loop arc
        arc = patches.Arc((nx, circle_y + 2.8), 3.2, 4.0, angle=0, theta1=25, theta2=155,
                          color=ACCENT_PURPLE, linewidth=0.7)
        ax.add_patch(arc)

        # Node Circle
        circ = Circle((nx, circle_y), 3.0, facecolor="#F8FAFC", edgecolor=ACCENT_PURPLE, linewidth=1.0, zorder=4)
        ax.add_patch(circ)
        ax.text(nx, circle_y, f"$v_{{{idx+1}}}$", ha="center", va="center", fontsize=6.2, fontweight="bold", color=ACCENT_PURPLE, zorder=5)

        # Bidirectional Edges to Adjacent Node
        if idx < len(node_xs) - 1:
            nxt = node_xs[idx + 1]
            ax.plot([nx + 3.2, nxt - 3.2], [circle_y, circle_y], color=BORDER_DARK, lw=0.9, zorder=2)

    ax.text(50, circle_y - 6.5, "Bidirectional Adjacency + Self-Loops (Feature dim = 768)", ha="center", fontsize=6.5, color=DARK_SLATE)

    draw_ieee_arrow(ax, (50, 41), (50, 32), lw=0.8)

    # Step 4: Mathematical Attention Callout
    draw_ieee_card(ax, 4, 4, 92, 25,
                   r"$\mathbf{4\text{-}Head\ GAT:}\ \alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_j]))}"
                   r"{\sum_{k \in \mathcal{N}_i} \exp(\mathrm{LeakyReLU}(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_k]))}$",
                   r"$\mathbf{Global\ Pooling:}\ H_{GAT} = \frac{1}{|V|} \sum_{v_i \in V} h_i' \in \mathbb{R}^{B \times 128}$",
                   bg="#FAF5FF", border=ACCENT_PURPLE, title_col=CHARCOAL, sub_col=ACCENT_PURPLE,
                   lw=0.9, title_sz=7.0, sub_sz=6.8)

    plt.tight_layout(pad=0.02)
    save_path = os.path.join(FIGURES_DIR, "gat_graph_construction.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


# ==============================================================================
# FIGURE 4: Dimension Compression Funnel (Single-Column: 3.5 x 2.0 in)
# ==============================================================================
def render_mlp_funnel_figure():
    fig, ax = plt.subplots(figsize=(3.5, 2.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(2, 94, "Projection MLP Dimension Compression", fontsize=8.2, fontweight="bold", color=CHARCOAL)
    ax.text(2, 88, "Harmonizing 960-d Multi-Perspective Embeddings into Z (128-d)", fontsize=6.8, color=MID_GRAY)

    # Stepped Funnel Blocks (Left to Right)
    stages = [
        {"x": 2,  "y": 14, "w": 20, "h": 68, "title": "Concat\n960-d", "sub": "768 BERT\n64 CNN\n128 GAT", "bg": "#FFFBEB", "border": "#B45309"},
        {"x": 27, "y": 20, "w": 21, "h": 56, "title": "Dense 1\n512-d", "sub": "BN + ReLU\np = 0.3", "bg": "#F0F7FF", "border": ACCENT_BLUE},
        {"x": 53, "y": 26, "w": 21, "h": 44, "title": "Dense 2\n256-d", "sub": "BN + ReLU\np = 0.2", "bg": "#F0FDF4", "border": ACCENT_TEAL},
        {"x": 79, "y": 32, "w": 19, "h": 32, "title": "Output\n128-d", "sub": "Latent Z\n-> XGBoost", "bg": "#FAF5FF", "border": ACCENT_PURPLE}
    ]

    for idx, s in enumerate(stages):
        draw_ieee_card(ax, s["x"], s["y"], s["w"], s["h"], s["title"], s["sub"],
                       bg=s["bg"], border=s["border"], lw=0.9, title_sz=7.2, sub_sz=6.2)
        if idx < len(stages) - 1:
            nxt_x = stages[idx + 1]["x"]
            mid_y = 48
            draw_ieee_arrow(ax, (s["x"] + s["w"] + 0.5, mid_y), (nxt_x - 0.5, mid_y), lw=0.9)

    # Bottom Technical Note
    ax.text(50, 5, "Xavier Uniform Initialization | Total Trainable Parameters = 657,792",
            ha="center", fontsize=6.4, color=MID_GRAY)

    plt.tight_layout(pad=0.02)
    save_path = os.path.join(FIGURES_DIR, "mlp_fusion_funnel.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


# ==============================================================================
# FIGURE 5: Causal Fidelity Verification Protocol (Single-Column: 3.5 x 2.2 in)
# ==============================================================================
def render_perturbation_figure():
    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(2, 94, "Causal Perturbation Verification Protocol", fontsize=8.2, fontweight="bold", color=CHARCOAL)
    ax.text(2, 88, "Targeted Feature Intervention vs. Random Masking Control", fontsize=6.8, color=MID_GRAY)

    # 1. Baseline Box
    draw_ieee_card(ax, 28, 66, 44, 18, "Unperturbed Input", r"Latent $Z \in \mathbb{R}^{128} \rightarrow \hat{p}_{\mathrm{base}} = 0.98$",
                   bg="#F8FAFC", border=BORDER_DARK, lw=0.9, title_sz=7.5, sub_sz=6.8)

    # Forking Arrows
    draw_ieee_arrow(ax, (40, 66), (22, 57), color=ACCENT_CRIMSON, lw=1.0)
    draw_ieee_arrow(ax, (60, 66), (78, 57), color=MID_GRAY, lw=1.0)

    # 2. Arm A: Targeted Intervention
    draw_ieee_card(ax, 2, 33, 44, 23, "Arm A: Top-10 SHAP Mask",
                   r"Replace with $\bar{z}_i \rightarrow \hat{p}_{\mathrm{pert}} = 0.34$" + "\n" + r"$\mathbf{\Delta p = 0.64}$ (Catastrophic Drop)",
                   bg="#FFF5F5", border=ACCENT_CRIMSON, title_col=ACCENT_CRIMSON, lw=1.0, title_sz=7.2, sub_sz=6.4)

    # 3. Arm B: Negative Control
    draw_ieee_card(ax, 54, 33, 44, 23, "Arm B: Random-10 Mask",
                   r"Replace with $\bar{z}_i \rightarrow \hat{p}_{\mathrm{pert}} = 0.93$" + "\n" + r"$\mathbf{\Delta p = 0.05}$ (Preserved Signal)",
                   bg="#F1F5F9", border=BORDER_DARK, title_col=DARK_SLATE, lw=0.9, title_sz=7.2, sub_sz=6.4)

    # Converging Arrows to Callout
    draw_ieee_arrow(ax, (24, 33), (38, 23), color=ACCENT_CRIMSON, lw=1.0)
    draw_ieee_arrow(ax, (76, 33), (62, 23), color=MID_GRAY, lw=1.0)

    # 4. Statistical Validation Box
    draw_ieee_card(ax, 10, 4, 80, 18,
                   r"$\mathbf{Effect\ Size:}\ d = \frac{\mu_{\mathrm{SHAP}} - \mu_{\mathrm{rand}}}{s_{\mathrm{pool}}} = 1.42\ (p < 0.001)$",
                   "Confirms causal biological necessity beyond passive correlation",
                   bg="#F0FDF4", border=ACCENT_GREEN, title_col=ACCENT_GREEN, sub_col=DARK_SLATE,
                   lw=1.0, title_sz=7.2, sub_sz=6.4)

    plt.tight_layout(pad=0.02)
    save_path = os.path.join(FIGURES_DIR, "perturbation_protocol.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 68)
    print("  AvianNet: Rendering IEEE Compact Publication Figures (300 DPI)")
    print("=" * 68)
    render_architecture_figure()
    render_trifold_xai_figure()
    render_gat_graph_figure()
    render_mlp_funnel_figure()
    render_perturbation_figure()
    print("=" * 68)
    print(f"  All 5 diagrams rendered into './{FIGURES_DIR}/'")
    print("=" * 68)
