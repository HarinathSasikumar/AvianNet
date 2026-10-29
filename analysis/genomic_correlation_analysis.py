"""
Genomic Correlation & K-mer Statistical Analysis
==================================================
Comprehensive inter-species DNA similarity analysis across all 6 avian species:
  - Gallus gallus (Chicken)
  - Anas platyrhynchos (Mallard Duck)
  - Meleagris gallopavo (Wild Turkey)
  - Anser anser (Greylag Goose)
  - Columba livia (Rock Pigeon)
  - Passer domesticus (House Sparrow)

Analyses performed:
  1. K-mer frequency profiles (3-mer and 4-mer)
  2. Inter-species Pearson & Spearman correlation matrix
  3. PCA of k-mer space (healthy vs infected clusters)
  4. GC content & nucleotide composition comparison
  5. Top shared/discriminative k-mers (common susceptibility signal)
  6. Jensen-Shannon divergence heatmap (sequence distribution similarity)
  7. Hierarchical clustering dendrogram
  8. Influenza-associated k-mer enrichment test
  9. Sequence length distributions per species
  10. Statistical summary report
"""

import sys, json, re, warnings
from pathlib import Path
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "raw"
OUT_DIR    = ROOT / "outputs" / "genomic_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
ACCENT    = "#06b6d4"
SUCCESS   = "#10b981"
WARNING   = "#f59e0b"
DANGER    = "#f43f5e"
PURPLE    = "#8b5cf6"
PALETTE   = [ACCENT, SUCCESS, WARNING, DANGER, PURPLE, "#fb923c"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "text.color":        "#e2e8f0",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        "#1e3a5f",
    "grid.alpha":        0.4,
    "axes.grid":         True,
    "font.family":       "DejaVu Sans",
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#f8fafc",
    "axes.titlesize":    13,
})

# ─── Species Config ───────────────────────────────────────────────────────────
SPECIES = {
    "Gallus gallus":         {"short": "G.gallus",   "color": ACCENT,   "healthy": ["gallus_gallus/healthy.fasta","healthy/general/gallus_gallus_cds.fasta","healthy/general/gallus_gallus_gene.fasta","healthy/general/gallus_gallus_fragment.fasta","healthy/general/gallus_exon.fasta","healthy/general/gallus_gallus_cds_expanded.fasta","healthy/general/gallus_gene_fragment.fasta","healthy/general/gallus_gallus_exon.fasta"], "infected": ["gallus_gallus/infected.fasta","infected/H5/H5N1_sequences.fasta","infected/H5/H5N1_expanded.fasta","infected/H5/H5N2_sequences.fasta","infected/H5/H5N2_expanded.fasta","infected/H5/H5N6_sequences.fasta","infected/H5/H5N8_sequences.fasta","infected/H7/H7N9_sequences.fasta","infected/H7/H7N3_sequences.fasta","infected/H7/H7N7_sequences.fasta"]},
    "Anas platyrhynchos":    {"short": "A.platyr.",  "color": SUCCESS,  "healthy": ["anas_platyrhynchos/healthy.fasta","healthy/general/duck_anas_platyrhynchos.fasta"], "infected": ["anas_platyrhynchos/infected.fasta"]},
    "Meleagris gallopavo":   {"short": "M.gallop.",  "color": WARNING,  "healthy": ["meleagris_gallopavo/healthy.fasta","healthy/general/turkey_meleagris_gallopavo.fasta"], "infected": []},
    "Anser anser":           {"short": "A.anser",    "color": DANGER,   "healthy": ["anser_anser/healthy.fasta"], "infected": []},
    "Columba livia":         {"short": "C.livia",    "color": PURPLE,   "healthy": ["columba_livia/healthy.fasta","healthy/general/pigeon_columba_livia.fasta"], "infected": ["columba_livia/infected.fasta"]},
    "Passer domesticus":     {"short": "P.domest.",  "color": "#fb923c","healthy": ["passer_domesticus/healthy.fasta"], "infected": []},
}

# ─── Parser ───────────────────────────────────────────────────────────────────
VALID_NT = set("ATGCN")

def parse_fasta(path: Path) -> list:
    seqs = []
    cur  = []
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur).upper())
                    cur = []
                else:
                    cleaned = re.sub(r"[^ATGCN]", "", line.upper())
                    if cleaned:
                        cur.append(cleaned)
        if cur:
            seqs.append("".join(cur).upper())
    except Exception:
        pass
    return [s for s in seqs if len(s) >= 20]


def load_species_sequences(paths: list, label: int, max_per_file=300) -> list:
    seqs = []
    for rel in paths:
        p = DATA_DIR / rel
        if p.exists():
            loaded = parse_fasta(p)[:max_per_file]
            seqs.extend([(s, label) for s in loaded])
    return seqs


# ─── K-mer Engine ─────────────────────────────────────────────────────────────
def kmer_freq(seq: str, k: int) -> dict:
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1)
                     if all(c in "ATGC" for c in seq[i:i+k]))
    total  = sum(counts.values()) or 1
    return {kmer: cnt/total for kmer, cnt in counts.items()}


def all_kmers(k: int) -> list:
    return ["".join(p) for p in product("ATGC", repeat=k)]


def seq_to_vector(seq: str, k: int, kmer_list: list) -> np.ndarray:
    freq = kmer_freq(seq, k)
    return np.array([freq.get(km, 0.0) for km in kmer_list], dtype=float)


def gc_content(seq: str) -> float:
    gc = seq.count("G") + seq.count("C")
    total = sum(seq.count(n) for n in "ATGC") or 1
    return gc / total * 100


# ─── Load All Data ─────────────────────────────────────────────────────────────
print("Loading FASTA sequences for all 6 species ...")
species_data = {}   # {species_name: {"healthy": [seq,...], "infected": [seq,...]}}

for sp, cfg in SPECIES.items():
    h = [s for s,l in load_species_sequences(cfg["healthy"], 0)]
    i = [s for s,l in load_species_sequences(cfg["infected"], 1)]
    species_data[sp] = {"healthy": h, "infected": i}
    total = len(h) + len(i)
    print(f"  {cfg['short']:12s}  healthy={len(h):4d}  infected={len(i):4d}  total={total}")

# ─── Build K-mer Matrices ─────────────────────────────────────────────────────
K3 = 3
K4 = 4
kmers3 = all_kmers(K3)
kmers4 = all_kmers(K4)

print("\nComputing 3-mer and 4-mer frequency matrices ...")

# Per-species mean k-mer vectors (healthy vs infected)
sp_kmer3_healthy   = {}
sp_kmer3_infected  = {}
sp_kmer4_healthy   = {}
sp_kmer4_infected  = {}

rows_3  = []  # for correlation matrix
row_meta = []

for sp, data in species_data.items():
    cfg = SPECIES[sp]
    # Healthy mean
    if data["healthy"]:
        vecs = np.array([seq_to_vector(s, K3, kmers3) for s in data["healthy"][:200]])
        sp_kmer3_healthy[sp] = vecs.mean(axis=0)
        rows_3.append(sp_kmer3_healthy[sp])
        row_meta.append({"species": sp, "label": "Healthy"})

        vecs4 = np.array([seq_to_vector(s, K4, kmers4) for s in data["healthy"][:200]])
        sp_kmer4_healthy[sp] = vecs4.mean(axis=0)
    else:
        sp_kmer3_healthy[sp] = np.zeros(len(kmers3))
        sp_kmer4_healthy[sp] = np.zeros(len(kmers4))

    if data["infected"]:
        vecs = np.array([seq_to_vector(s, K3, kmers3) for s in data["infected"][:200]])
        sp_kmer3_infected[sp] = vecs.mean(axis=0)
        rows_3.append(sp_kmer3_infected[sp])
        row_meta.append({"species": sp, "label": "Infected"})

        vecs4 = np.array([seq_to_vector(s, K4, kmers4) for s in data["infected"][:200]])
        sp_kmer4_infected[sp] = vecs4.mean(axis=0)
    else:
        sp_kmer3_infected[sp] = np.zeros(len(kmers3))
        sp_kmer4_infected[sp] = np.zeros(len(kmers4))

print("  Done.")

# ─── 1. CORRELATION MATRIX (species-level 3-mer, healthy only) ────────────────
print("Plot 1: Inter-species 3-mer Pearson correlation heatmap ...")

sp_names = list(SPECIES.keys())
sp_shorts = [SPECIES[s]["short"] for s in sp_names]
n = len(sp_names)

corr_mat_pearson  = np.zeros((n, n))
corr_mat_spearman = np.zeros((n, n))

for i, s1 in enumerate(sp_names):
    for j, s2 in enumerate(sp_names):
        v1, v2 = sp_kmer3_healthy[s1], sp_kmer3_healthy[s2]
        if np.any(v1) and np.any(v2):
            corr_mat_pearson[i, j],  _ = pearsonr(v1, v2)
            corr_mat_spearman[i, j], _ = spearmanr(v1, v2)
        else:
            corr_mat_pearson[i, j] = corr_mat_spearman[i, j] = 0.0

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Inter-Species 3-mer Correlation (Healthy Sequences)", fontsize=16, fontweight="bold", color="#f8fafc")

for ax, mat, title, cmap in zip(axes, [corr_mat_pearson, corr_mat_spearman],
                                  ["Pearson Correlation", "Spearman Rank Correlation"],
                                  ["YlOrRd", "PuBuGn"]):
    im = ax.imshow(mat, cmap=cmap, vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(sp_shorts, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(sp_shorts, fontsize=11)
    ax.set_title(title, fontsize=13)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="black" if mat[i,j] > 0.9 else "white")
    fig.colorbar(im, ax=ax, label="Correlation Coefficient", pad=0.02)

fig.tight_layout()
fig.savefig(OUT_DIR / "01_correlation_matrix.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 01_correlation_matrix.png")

# ─── 2. JS DIVERGENCE HEATMAP ─────────────────────────────────────────────────
print("Plot 2: Jensen-Shannon divergence heatmap ...")

def safe_js(v1, v2):
    s = v1 + v2
    if s.sum() < 1e-12:
        return 1.0
    p = v1 / (v1.sum() + 1e-12)
    q = v2 / (v2.sum() + 1e-12)
    return float(jensenshannon(p, q))

js_mat = np.zeros((n, n))
for i, s1 in enumerate(sp_names):
    for j, s2 in enumerate(sp_names):
        js_mat[i, j] = safe_js(sp_kmer3_healthy[s1], sp_kmer3_healthy[s2])

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Jensen-Shannon Divergence Between Species (3-mer Distributions)", fontsize=14, fontweight="bold", color="#f8fafc")
cmap_js = LinearSegmentedColormap.from_list("js", ["#10b981", "#f59e0b", "#f43f5e"])
im = ax.imshow(js_mat, cmap=cmap_js, vmin=0, vmax=0.3)
ax.set_xticks(range(n)); ax.set_xticklabels(sp_shorts, rotation=30, ha="right", fontsize=11)
ax.set_yticks(range(n)); ax.set_yticklabels(sp_shorts, fontsize=11)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{js_mat[i,j]:.4f}", ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if js_mat[i,j] > 0.15 else "black")
fig.colorbar(im, ax=ax, label="JSD (0=identical, 1=completely different)", pad=0.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_js_divergence_heatmap.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 02_js_divergence_heatmap.png")

# ─── 3. HIERARCHICAL DENDROGRAM ───────────────────────────────────────────────
print("Plot 3: Hierarchical clustering dendrogram ...")

dist_vec = []
labels_d = []
for sp in sp_names:
    dist_vec.append(sp_kmer3_healthy[sp])
    labels_d.append(SPECIES[sp]["short"] + "\n(Healthy)")
    if np.any(sp_kmer3_infected[sp]):
        dist_vec.append(sp_kmer3_infected[sp])
        labels_d.append(SPECIES[sp]["short"] + "\n(Infected)")

dist_vec = np.array(dist_vec)
Z = linkage(dist_vec, method="ward", metric="euclidean")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)
fig.suptitle("Hierarchical Clustering — Ward Linkage on 3-mer k-mer Space", fontsize=14, fontweight="bold", color="#f8fafc")
dendrogram(Z, labels=labels_d, ax=ax, leaf_font_size=9,
           color_threshold=0.5 * max(Z[:,2]),
           above_threshold_color="#94a3b8")
ax.set_ylabel("Euclidean Distance (k-mer space)", fontsize=11)
ax.tick_params(axis="x", colors="#e2e8f0")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_dendrogram.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 03_dendrogram.png")

# ─── 4. PCA — ALL SEQUENCES COLORED BY SPECIES & LABEL ───────────────────────
print("Plot 4: PCA scatter — species and health status ...")

all_vecs, all_sp, all_labels = [], [], []
for sp, data in species_data.items():
    for seq in data["healthy"][:80]:
        all_vecs.append(seq_to_vector(seq, K3, kmers3))
        all_sp.append(sp); all_labels.append("Healthy")
    for seq in data["infected"][:80]:
        all_vecs.append(seq_to_vector(seq, K3, kmers3))
        all_sp.append(sp); all_labels.append("Infected")

all_vecs = np.array(all_vecs)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(all_vecs)
pca      = PCA(n_components=3, random_state=42)
X_pca    = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"PCA of 3-mer K-mer Space — Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%, PC3={pca.explained_variance_ratio_[2]*100:.1f}%",
             fontsize=13, fontweight="bold", color="#f8fafc")

# Left: Colored by species
ax = axes[0]
for idx, sp in enumerate(sp_names):
    mask = [i for i, s in enumerate(all_sp) if s == sp]
    if mask:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=SPECIES[sp]["color"], label=SPECIES[sp]["short"],
                   alpha=0.7, s=40, edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax.set_title("Colored by Species", fontsize=13)
ax.legend(fontsize=9, framealpha=0.3, loc="best")

# Right: Colored by Healthy/Infected
ax = axes[1]
for lbl, col in [("Healthy", SUCCESS), ("Infected", DANGER)]:
    mask = [i for i, l in enumerate(all_labels) if l == lbl]
    if mask:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=col, label=lbl, alpha=0.7, s=40, edgecolors="none",
                   marker="o" if lbl == "Healthy" else "^")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax.set_title("Colored by Health Status", fontsize=13)
ax.legend(fontsize=11, framealpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "04_pca_scatter.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 04_pca_scatter.png")

# ─── 5. GC CONTENT & NUCLEOTIDE COMPOSITION ───────────────────────────────────
print("Plot 5: GC content and nucleotide composition ...")

comp_data = []
for sp, data in species_data.items():
    for seq in data["healthy"][:150]:
        total = sum(seq.count(n) for n in "ATGC") or 1
        comp_data.append({
            "Species": SPECIES[sp]["short"], "Label": "Healthy",
            "GC%": gc_content(seq),
            "A%": seq.count("A")/total*100, "T%": seq.count("T")/total*100,
            "G%": seq.count("G")/total*100, "C%": seq.count("C")/total*100,
            "Length": len(seq)
        })
    for seq in data["infected"][:150]:
        total = sum(seq.count(n) for n in "ATGC") or 1
        comp_data.append({
            "Species": SPECIES[sp]["short"], "Label": "Infected",
            "GC%": gc_content(seq),
            "A%": seq.count("A")/total*100, "T%": seq.count("T")/total*100,
            "G%": seq.count("G")/total*100, "C%": seq.count("C")/total*100,
            "Length": len(seq)
        })

df = pd.DataFrame(comp_data)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Nucleotide Composition & GC Content by Species and Health Status", fontsize=15, fontweight="bold", color="#f8fafc")

# GC% by species
ax = axes[0, 0]
order = [SPECIES[s]["short"] for s in sp_names]
sns.boxplot(data=df, x="Species", y="GC%", hue="Label",
            palette={"Healthy": SUCCESS, "Infected": DANGER},
            ax=ax, order=order, width=0.6, linewidth=1.2)
ax.set_title("GC Content Distribution (%) by Species")
ax.set_xlabel(""); ax.legend(fontsize=10, framealpha=0.3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

# Nucleotide composition bar chart
ax = axes[0, 1]
sp_gc_means = df.groupby(["Species","Label"])["GC%"].mean().reset_index()
colors_sp = [SPECIES[s]["color"] for s in sp_names]
sp_shorts_u = [SPECIES[s]["short"] for s in sp_names]
nt_means = []
for sp in sp_names:
    h = df[df["Species"]==SPECIES[sp]["short"]]
    if not h.empty:
        nt_means.append([h["A%"].mean(), h["T%"].mean(), h["G%"].mean(), h["C%"].mean()])
    else:
        nt_means.append([0,0,0,0])

nt_means = np.array(nt_means)
x = np.arange(n)
w = 0.18
nts = ["A", "T", "G", "C"]
nt_cols = ["#22d3ee", "#a78bfa", "#34d399", "#fb923c"]
for ni, (nt, col) in enumerate(zip(nts, nt_cols)):
    ax.bar(x + (ni-1.5)*w, nt_means[:, ni], w, label=nt, color=col, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(sp_shorts_u, rotation=20, ha="right")
ax.set_ylabel("Mean Frequency (%)"); ax.set_title("Nucleotide Composition per Species (Healthy)")
ax.legend(fontsize=10, framealpha=0.3)

# Sequence length violin plot
ax = axes[1, 0]
log_df = df.copy()
log_df["Length"] = np.log10(log_df["Length"].clip(lower=1))
sns.violinplot(data=log_df, x="Species", y="Length", hue="Label",
               palette={"Healthy": SUCCESS, "Infected": DANGER},
               ax=ax, order=order, inner="quartile", linewidth=1)
ax.set_title("Sequence Length Distribution (log10 scale)")
ax.set_xlabel(""); ax.set_ylabel("log10(Length)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend(fontsize=10, framealpha=0.3)

# GC% scatter — healthy vs infected per species
ax = axes[1, 1]
for sp in sp_names:
    s = SPECIES[sp]["short"]
    h = df[(df["Species"]==s) & (df["Label"]=="Healthy")]["GC%"]
    i = df[(df["Species"]==s) & (df["Label"]=="Infected")]["GC%"]
    ax.scatter([s]*len(h), h, color=SPECIES[sp]["color"], alpha=0.4, s=15, label=None)
    ax.scatter([s]*len(i), i, color=DANGER, alpha=0.6, s=20, marker="^", label=None)
ax.set_title("GC% Spread — Circles=Healthy, Triangles=Infected")
ax.set_ylabel("GC Content (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right") if ax.get_xticklabels() else None

fig.tight_layout()
fig.savefig(OUT_DIR / "05_gc_nucleotide_composition.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 05_gc_nucleotide_composition.png")

# ─── 6. TOP SHARED K-MERS ACROSS ALL SPECIES (susceptibility signal) ──────────
print("Plot 6: Shared top k-mers and influenza susceptibility k-mers ...")

# Known influenza-associated k-mers from literature (HA gene receptor-binding domain)
INFLUENZA_KMERS = {
    "ATG": "Start codon / HA initiation",
    "GGC": "Glycine-rich HA stem region",
    "CAA": "Glutamine — receptor binding pocket",
    "TGC": "Cysteine bridge (NA active site)",
    "GCC": "Structural stability PB2 gene",
    "AAG": "Lysine — M2 proton channel",
    "CGT": "Arginine cleavage site motif",
    "TCA": "Serine phosphorylation site",
    "AGT": "Serine-rich domain",
    "GAT": "Aspartate ion transport"
}

# Average 3-mer frequencies across all healthy species
all_healthy_mean = {}
for km in kmers3:
    vals = [sp_kmer3_healthy[sp][kmers3.index(km)] for sp in sp_names if np.any(sp_kmer3_healthy[sp])]
    all_healthy_mean[km] = np.mean(vals) if vals else 0.0

# Top 20 most shared k-mers
top20 = sorted(all_healthy_mean, key=all_healthy_mean.get, reverse=True)[:20]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Shared K-mer Profiles Across All 6 Avian Species", fontsize=15, fontweight="bold", color="#f8fafc")

# Left: top 20 shared 3-mers
ax = axes[0]
vals = [all_healthy_mean[km] * 100 for km in top20]
colors_bar = [DANGER if km in INFLUENZA_KMERS else ACCENT for km in top20]
bars = ax.barh(top20[::-1], vals[::-1], color=colors_bar[::-1], edgecolor="none", alpha=0.9)
ax.set_xlabel("Mean Frequency (%) across all species", fontsize=11)
ax.set_title("Top 20 Most Abundant 3-mers (Red = Influenza-Associated)")
ax.axvline(x=np.mean(vals), color=WARNING, lw=1.5, linestyle="--", label=f"Mean: {np.mean(vals):.2f}%")
ax.legend(fontsize=10, framealpha=0.3)

# Right: per-species frequency of influenza-associated k-mers
ax = axes[1]
inf_kmer_list = list(INFLUENZA_KMERS.keys())
sp_inf_vals = {}
for sp in sp_names:
    sp_inf_vals[sp] = [sp_kmer3_healthy[sp][kmers3.index(km)] * 100 if km in kmers3 else 0.0 for km in inf_kmer_list]

x = np.arange(len(inf_kmer_list))
width = 0.13
for idx, sp in enumerate(sp_names):
    ax.bar(x + (idx - 2.5) * width, sp_inf_vals[sp], width,
           label=SPECIES[sp]["short"], color=SPECIES[sp]["color"], alpha=0.85, edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(inf_kmer_list, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Frequency (%) in Healthy Sequences")
ax.set_title("Influenza-Associated K-mer Frequency Across All 6 Species")
ax.legend(fontsize=8, framealpha=0.3, ncol=2)

fig.tight_layout()
fig.savefig(OUT_DIR / "06_shared_kmer_susceptibility.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 06_shared_kmer_susceptibility.png")

# ─── 7. HEALTHY vs INFECTED K-MER SHIFT (per species) ─────────────────────────
print("Plot 7: Healthy vs infected k-mer shift analysis ...")

sp_with_infected = [sp for sp in sp_names if np.any(sp_kmer3_infected[sp])]
top_kmers_shift = sorted(kmers3, key=lambda km: abs(
    np.mean([sp_kmer3_infected[sp][kmers3.index(km)] - sp_kmer3_healthy[sp][kmers3.index(km)]
             for sp in sp_with_infected])
), reverse=True)[:15]

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("3-mer Frequency Shift: Infected − Healthy (Top 15 Discriminative K-mers)", fontsize=14, fontweight="bold", color="#f8fafc")

x = np.arange(len(top_kmers_shift))
w = 0.18
for idx, sp in enumerate(sp_with_infected):
    shifts = [(sp_kmer3_infected[sp][kmers3.index(km)] - sp_kmer3_healthy[sp][kmers3.index(km)]) * 100
              for km in top_kmers_shift]
    ax.bar(x + (idx - len(sp_with_infected)/2 + 0.5) * w, shifts, w,
           label=SPECIES[sp]["short"], color=SPECIES[sp]["color"], alpha=0.85, edgecolor="none")

ax.axhline(0, color="#94a3b8", lw=1.2, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(top_kmers_shift, rotation=30, ha="right", fontsize=11)
ax.set_ylabel("Frequency Shift (%): Infected − Healthy")
ax.legend(fontsize=10, framealpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "07_kmer_shift_infected_vs_healthy.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 07_kmer_shift_infected_vs_healthy.png")

# ─── 8. 4-MER HEATMAP (species profiles) ─────────────────────────────────────
print("Plot 8: 4-mer heatmap per species ...")

top_4mers = sorted(kmers4, key=lambda km: np.std([sp_kmer4_healthy[sp][kmers4.index(km)] for sp in sp_names]), reverse=True)[:40]

mat4 = np.array([[sp_kmer4_healthy[sp][kmers4.index(km)] * 100 for km in top_4mers] for sp in sp_names])

fig, ax = plt.subplots(figsize=(20, 6))
fig.suptitle("Top 40 Most Variable 4-mer K-mer Profiles Across Species (Healthy)", fontsize=14, fontweight="bold", color="#f8fafc")
im = ax.imshow(mat4, cmap="viridis", aspect="auto")
ax.set_yticks(range(n)); ax.set_yticklabels(sp_shorts)
ax.set_xticks(range(len(top_4mers))); ax.set_xticklabels(top_4mers, rotation=45, ha="right", fontsize=8)
fig.colorbar(im, ax=ax, label="Frequency (%)", pad=0.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "08_4mer_species_heatmap.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 08_4mer_species_heatmap.png")

# ─── 9. STATISTICAL TEST — ANOVA on GC% across species ───────────────────────
print("Plot 9: ANOVA and pairwise t-test on GC% ...")

gc_groups = {}
for sp in sp_names:
    s = SPECIES[sp]["short"]
    gc_groups[s] = df[df["Species"]==s]["GC%"].dropna().values

f_stat, p_val = stats.f_oneway(*[gc_groups[s] for s in gc_groups if len(gc_groups[s]) > 1])
print(f"  ANOVA: F={f_stat:.4f}, p={p_val:.4e}")

# Pairwise matrix
short_list = list(gc_groups.keys())
nn = len(short_list)
pval_mat = np.ones((nn, nn))
for i, s1 in enumerate(short_list):
    for j, s2 in enumerate(short_list):
        if i != j and len(gc_groups[s1]) > 1 and len(gc_groups[s2]) > 1:
            _, pv = stats.ttest_ind(gc_groups[s1], gc_groups[s2], equal_var=False)
            pval_mat[i, j] = pv

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"Statistical Tests on GC Content  |  ANOVA: F={f_stat:.2f}, p={p_val:.2e}", fontsize=14, fontweight="bold", color="#f8fafc")

# Left: pairwise p-value matrix
ax = axes[0]
neg_log_p = -np.log10(pval_mat + 1e-300)
im = ax.imshow(neg_log_p, cmap="hot", aspect="auto")
ax.set_xticks(range(nn)); ax.set_xticklabels(short_list, rotation=30, ha="right", fontsize=10)
ax.set_yticks(range(nn)); ax.set_yticklabels(short_list, fontsize=10)
for i in range(nn):
    for j in range(nn):
        pv = pval_mat[i, j]
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        ax.text(j, i, sig, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white" if neg_log_p[i, j] > 1 else "black")
fig.colorbar(im, ax=ax, label="-log10(p-value)", pad=0.02)
ax.set_title("Pairwise Welch's t-test on GC% (* p<0.05, ** p<0.01, *** p<0.001)")

# Right: mean GC% with error bars
ax = axes[1]
means = [gc_groups[s].mean() if len(gc_groups[s]) > 0 else 0 for s in short_list]
stds  = [gc_groups[s].std()  if len(gc_groups[s]) > 0 else 0 for s in short_list]
colors_bar = [SPECIES[sp]["color"] for sp in sp_names]
bars = ax.bar(range(nn), means, yerr=stds, color=colors_bar, capsize=5,
              error_kw={"ecolor":"#94a3b8","elinewidth":1.5}, edgecolor="none", alpha=0.9)
ax.set_xticks(range(nn)); ax.set_xticklabels(short_list, rotation=20, ha="right")
ax.set_ylabel("Mean GC Content (%)"); ax.set_title("Mean GC% per Species (±1 SD)")
ax.axhline(np.nanmean(means), color=WARNING, lw=1.5, linestyle="--", label=f"Grand mean: {np.nanmean(means):.1f}%")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.5, f"{m:.1f}%", ha="center", fontsize=9, color="#e2e8f0")
ax.legend(fontsize=10, framealpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "09_gc_anova_pairwise.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 09_gc_anova_pairwise.png")

# ─── 10. INFLUENZA K-MER ENRICHMENT: infected vs healthy per species ──────────
print("Plot 10: Influenza k-mer enrichment analysis ...")

enrich_rows = []
for sp in sp_with_infected:
    for km in list(INFLUENZA_KMERS.keys()):
        idx_km = kmers3.index(km)
        h_freq = sp_kmer3_healthy[sp][idx_km]
        i_freq = sp_kmer3_infected[sp][idx_km]
        fold   = (i_freq / (h_freq + 1e-9))
        enrich_rows.append({"Species": SPECIES[sp]["short"], "k-mer": km, "Healthy": h_freq*100, "Infected": i_freq*100, "Fold Change": fold, "Function": INFLUENZA_KMERS[km]})

edf = pd.DataFrame(enrich_rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Influenza-Associated K-mer Enrichment in Infected vs Healthy Sequences", fontsize=14, fontweight="bold", color="#f8fafc")

# Left: fold change heatmap
ax = axes[0]
pivot = edf.pivot(index="k-mer", columns="Species", values="Fold Change")
cmap_fc = LinearSegmentedColormap.from_list("fc", ["#3b82f6", "#f8fafc", "#f43f5e"])
im = ax.imshow(pivot.values, cmap=cmap_fc, vmin=0.5, vmax=2.0, aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=10)
ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=11)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=9, fontweight="bold",
                color="black" if 0.8 < v < 1.2 else "white")
fig.colorbar(im, ax=ax, label="Fold Change (Infected/Healthy)", pad=0.02)
ax.set_title("Fold Change Heatmap per Species and K-mer")

# Right: scatter plot healthy vs infected frequency
ax = axes[1]
sp_col_map = {SPECIES[sp]["short"]: SPECIES[sp]["color"] for sp in sp_names}
for _, row in edf.iterrows():
    col = sp_col_map.get(row["Species"], ACCENT)
    ax.scatter(row["Healthy"], row["Infected"], color=col, s=80, alpha=0.8, edgecolors="none")
    if row["k-mer"] in ["ATG", "GGC", "CAA", "TGC"]:
        ax.annotate(row["k-mer"], (row["Healthy"], row["Infected"]),
                    fontsize=8, color="#e2e8f0", xytext=(3, 3), textcoords="offset points")

ax.plot([0, edf[["Healthy","Infected"]].max().max()],
        [0, edf[["Healthy","Infected"]].max().max()],
        "--", color="#94a3b8", lw=1.2, label="y=x (no change)")
for sp in sp_with_infected:
    ax.scatter([], [], color=SPECIES[sp]["color"], label=SPECIES[sp]["short"], s=60)
ax.set_xlabel("Healthy Frequency (%)")
ax.set_ylabel("Infected Frequency (%)")
ax.set_title("Healthy vs Infected Frequency of Influenza K-mers")
ax.legend(fontsize=9, framealpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "10_influenza_enrichment.png", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("  Saved: 10_influenza_enrichment.png")

# ─── SUMMARY STATISTICS JSON ──────────────────────────────────────────────────
print("\nGenerating JSON summary ...")

summary = {
    "analysis": "Genomic Correlation & K-mer Statistical Analysis",
    "species_count": 6,
    "per_species_stats": {},
    "correlation_findings": {},
    "js_divergence": {},
    "influenza_kmer_findings": {},
    "anova_gc": {"F_statistic": float(f_stat), "p_value": float(p_val), "significant": bool(p_val < 0.05)},
    "pca_explained_variance": {f"PC{i+1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_[:3])},
}

for sp in sp_names:
    s = SPECIES[sp]["short"]
    gc_h = gc_groups.get(s, np.array([0]))
    summary["per_species_stats"][sp] = {
        "short": s,
        "healthy_count": len(species_data[sp]["healthy"]),
        "infected_count": len(species_data[sp]["infected"]),
        "mean_gc_healthy": float(gc_h.mean()) if len(gc_h) > 0 else None,
        "std_gc_healthy": float(gc_h.std()) if len(gc_h) > 0 else None,
    }
    for sp2 in sp_names:
        summary["correlation_findings"][f"{SPECIES[sp]['short']}_vs_{SPECIES[sp2]['short']}"] = {
            "pearson": float(corr_mat_pearson[sp_names.index(sp), sp_names.index(sp2)]),
            "spearman": float(corr_mat_spearman[sp_names.index(sp), sp_names.index(sp2)]),
            "js_divergence": float(js_mat[sp_names.index(sp), sp_names.index(sp2)])
        }

for km, func in INFLUENZA_KMERS.items():
    summary["influenza_kmer_findings"][km] = {
        "function": func,
        "mean_freq_all_species_healthy_pct": float(all_healthy_mean.get(km, 0) * 100),
        "present_in_all_species": all(sp_kmer3_healthy[sp][kmers3.index(km)] > 0 for sp in sp_names)
    }

with open(OUT_DIR / "genomic_analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ─── PRINT FINAL REPORT ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("  GENOMIC CORRELATION ANALYSIS — COMPLETE SUMMARY")
print("="*70)
print(f"\n{'Species':<25} {'Healthy':>8} {'Infected':>9} {'Mean GC%':>10} {'SD GC%':>8}")
print("-"*65)
for sp in sp_names:
    s = SPECIES[sp]["short"]
    gc_h = gc_groups.get(s, np.array([0]))
    print(f"  {sp:<23} {len(species_data[sp]['healthy']):>8} {len(species_data[sp]['infected']):>9} {gc_h.mean():>10.2f} {gc_h.std():>8.2f}")

print(f"\nPCA Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%, PC3={pca.explained_variance_ratio_[2]*100:.1f}%")
print(f"\nANOVA on GC% across species: F={f_stat:.4f}, p={p_val:.4e} {'(SIGNIFICANT)' if p_val<0.05 else ''}")
print(f"\nMin Pearson correlation (between any two healthy species): {corr_mat_pearson[corr_mat_pearson>0].min():.4f}")
print(f"Max JSD  (most dissimilar pair): {js_mat[js_mat>0].max():.4f}")

print("\nInfluenza-Associated K-mers present in ALL 6 healthy species:")
common = [km for km in INFLUENZA_KMERS if all(sp_kmer3_healthy[sp][kmers3.index(km)] > 0 for sp in sp_names)]
for km in common:
    print(f"  {km}  — {INFLUENZA_KMERS[km]}  (mean freq: {all_healthy_mean[km]*100:.3f}%)")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("="*70)
