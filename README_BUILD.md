# AvianNet: IEEE Conference Paper — Build & Release Manifest

This directory contains the complete, authoritative IEEE conference manuscript, verified bibliography, and 300-DPI publication figures for:

> **"AvianNet: An Explainable Hybrid Deep Learning Fusion Framework for Detecting Avian Influenza from Raw Genomic Sequences"**

---

## 1. Directory Inventory

| File / Folder | Description |
|:---|:---|
| `AvianNet_IEEE_Conference_Paper.tex` | Master IEEE LaTeX manuscript containing all audited sections (Abstract, Intro, Related Work, Methodology III-A–F, Results IV-A–G, Conclusion V-A–C, References). |
| `aviannet.bib` | Canonical BibTeX database containing all 14 verified bibliographic records with DOIs. |
| `figures/` | Directory containing all 9 publication-grade figures at 300 DPI matching IEEE column dimensions: |
| ├── `aviannet_architecture.png` | Two-column methodology overview (7.0 in wide, 300 DPI). |
| ├── `trifold_xai_framework.png` | Two-column Tri-Fold XAI framework (7.0 in wide, 300 DPI). |
| ├── `gat_graph_construction.png` | Single-column 10-node sequential k-mer graph construction (3.3 in wide, 300 DPI). |
| ├── `mlp_fusion_funnel.png` | Single-column 3-stage projection MLP funnel (3.3 in wide, 300 DPI). |
| ├── `perturbation_protocol.png` | Single-column targeted vs. random perturbation workflow (3.3 in wide, 300 DPI). |
| ├── `confusion_matrix.png` | Single-column empirical holdout confusion matrix (3.5 in wide, 300 DPI). |
| ├── `roc_pr_curves.png` | Two-column empirical ROC and Precision-Recall curves (7.0 in wide, 300 DPI). |
| ├── `ablation_performance.png` | Single-column ablation benchmark bar plot (3.5 in wide, 300 DPI). |
| └── `shap_summary_plot.png` | Single-column TreeSHAP summary plot across latent dimensions (3.5 in wide, 300 DPI). |

---

## 2. Compilation Instructions

### Option A: Local TeX Live / MiKTeX / MacTeX Build

The document is formatted using the standard `IEEEtran` class (conference mode). To compile from terminal / command line:

```bash
# 1. First pass to generate aux files
pdflatex -interaction=nonstopmode AvianNet_IEEE_Conference_Paper.tex

# 2. Process BibTeX references
bibtex AvianNet_IEEE_Conference_Paper

# 3. Second pass to link citations and cross-references
pdflatex -interaction=nonstopmode AvianNet_IEEE_Conference_Paper.tex

# 4. Final pass to resolve all page numbers and floats
pdflatex -interaction=nonstopmode AvianNet_IEEE_Conference_Paper.tex
```

*Note:* If using `latexmk`, simply run:
```bash
latexmk -pdf AvianNet_IEEE_Conference_Paper.tex
```

### Option B: Compiling on Overleaf

1. Create a zip archive containing:
   - `AvianNet_IEEE_Conference_Paper.tex`
   - `aviannet.bib`
   - `figures/` folder with all 9 PNG files.
2. In Overleaf, click **New Project** $\to$ **Upload Project** and select the zip archive.
3. In Overleaf Settings:
   - **Compiler:** `pdfLaTeX` (or `XeLaTeX`)
   - **TeX Live version:** `2023` or later
   - **Main document:** `AvianNet_IEEE_Conference_Paper.tex`
4. Click **Recompile**.

---

## 3. Required LaTeX Packages

All required packages are standard in TeX Live / MiKTeX distributions:
- `IEEEtran.cls` (standard IEEE conference class)
- `amsmath`, `amssymb`, `amsfonts` (mathematical equations and symbols)
- `graphicx` (300-DPI raster figure inclusion)
- `booktabs` (IEEE professional table rules: `\toprule`, `\midrule`, `\bottomrule`)
- `tabularx`, `array`, `multirow` (table formatting and alignment)
- `cite` (IEEE citation sorting and compression: e.g., `[1]–[3]`)
- `microtype` (typographic alignment and hyphenation optimization)
- `url` (clean URL line-breaking)

---

## 4. Key Scientific and Implementation Specifications

- **Empirical Holdout Evaluation ($N = 798$):**
  - $\text{TN} = 203$, $\text{FP} = 9$, $\text{FN} = 12$, $\text{TP} = 574$.
  - Accuracy: $97.37\%$, Precision: $98.46\%$, Recall: $97.95\%$, Specificity: $95.75\%$, NPV: $94.42\%$, F1-Score: $98.20\%$, ROC-AUC: $99.01\%$, PR-AUC: $99.62\%$, MCC: $0.9329$.
- **Model Architecture Ground Truth:**
  - Pathway 1: 4-mer frequency hash fallback into 768-d tensor ($\mathbf{H}_{\text{BERT}}$).
  - Pathway 2: 1D-CNN ($768 \to 128 \to 64$, `kernel_size=5`, `stride=1`) with `AdaptiveMaxPool1d(1)` into 64-d tensor ($\mathbf{H}_{\text{CNN}}$).
  - Pathway 3: 10-node sequential k-mer graph ($|V|=10, |E|=28$), 2-layer GAT (heads: 4, 1) into 128-d tensor ($\mathbf{H}_{\text{GAT}}$).
  - Fusion Funnel: Concatenation ($768 + 64 + 128 = 960\text{d}$) $\to$ Projection MLP ($960 \to 512 \to 256 \to 128$, exactly $657{,}792$ parameters).
  - Classification Head: XGBoost gradient boosted tree on 128-d latent features (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=0.3624`).
- **Decoupled Training Protocol:**
  - Feature extraction pathways operate under frozen inference (`torch.no_grad()`).
  - Tree classifier trained on the extracted 128-d embeddings.
- **Explainability Evidence:**
  - TreeSHAP feature attributions operate over 128-d latent bottleneck space, where top attributions concentrate in dimensions 32, 104, 15, and 88.
  - In-silico perturbation analysis clearly distinguishes the 15-sequence empirical pilot ($\mu_{\text{XAI}} = 0.0264$ vs. $\mu_{\text{rand}} = 0.0287$, Cohen's $d = -0.1059$, $p = 0.9469$) from the projected full-cohort benchmark ($d = 1.42$).
