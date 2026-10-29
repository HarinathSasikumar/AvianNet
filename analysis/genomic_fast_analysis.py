"""
Fast Genomic K-mer Analysis — Single Script, optimized for speed.
Limits: max 50 sequences per file, max 300 chars per seq, no TSNE.
"""
import sys, json, re, warnings
from pathlib import Path
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
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
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "outputs" / "genomic_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
DARK="#0f172a"; CARD="#1e293b"; C1="#06b6d4"; C2="#10b981"; C3="#f59e0b"
C4="#f43f5e"; C5="#8b5cf6"; C6="#fb923c"
PAL=[C1,C2,C3,C4,C5,C6]
plt.rcParams.update({"figure.facecolor":DARK,"axes.facecolor":CARD,"axes.edgecolor":"#334155",
    "axes.labelcolor":"#94a3b8","text.color":"#e2e8f0","xtick.color":"#94a3b8",
    "ytick.color":"#94a3b8","grid.color":"#1e3a5f","grid.alpha":0.4,"axes.grid":True,
    "font.family":"DejaVu Sans","axes.titleweight":"bold","axes.titlecolor":"#f8fafc","axes.titlesize":12})

# ── Species ────────────────────────────────────────────────────────────
SPECIES = {
    "Gallus gallus":       {"s":"G.gallus", "c":C1,
        "h":["gallus_gallus/healthy.fasta","healthy/general/gallus_gallus_cds.fasta"],
        "i":["gallus_gallus/infected.fasta","infected/H5/H5N1_sequences.fasta","infected/H7/H7N9_sequences.fasta"]},
    "Anas platyrhynchos":  {"s":"A.platyr.", "c":C2,
        "h":["anas_platyrhynchos/healthy.fasta","healthy/general/duck_anas_platyrhynchos.fasta"],
        "i":["anas_platyrhynchos/infected.fasta"]},
    "Meleagris gallopavo": {"s":"M.gallop.","c":C3,
        "h":["meleagris_gallopavo/healthy.fasta","healthy/general/turkey_meleagris_gallopavo.fasta"],"i":[]},
    "Anser anser":         {"s":"A.anser",  "c":C4,
        "h":["anser_anser/healthy.fasta"],"i":[]},
    "Columba livia":       {"s":"C.livia",  "c":C5,
        "h":["columba_livia/healthy.fasta","healthy/general/pigeon_columba_livia.fasta"],
        "i":["columba_livia/infected.fasta"]},
    "Passer domesticus":   {"s":"P.domest.","c":C6,
        "h":["passer_domesticus/healthy.fasta"],"i":[]},
}
SP = list(SPECIES.keys())
SHORTS = [SPECIES[s]["s"] for s in SP]
COLORS = [SPECIES[s]["c"] for s in SP]
N = len(SP)

INFLUENZA_KMERS = {"ATG":"HA initiation codon","GGC":"Glycine-rich HA stem","CAA":"Glutamine receptor pocket",
    "TGC":"Cysteine bridge NA","GCC":"PB2 structural stability","AAG":"Lysine M2 channel",
    "CGT":"Arginine cleavage site","TCA":"Serine phosphorylation","AGT":"Serine-rich domain","GAT":"Aspartate transport"}

# ── Helpers ─────────────────────────────────────────────────────────────
def parse_fasta(path, maxn=50, maxlen=400):
    seqs, cur = [], []
    try:
        with open(path,"r",errors="ignore") as f:
            for line in f:
                l = line.strip()
                if not l: continue
                if l.startswith(">"):
                    if cur: seqs.append("".join(cur).upper())
                    if len(seqs) >= maxn: break
                    cur = []
                else:
                    cur.append(re.sub(r"[^ATGCN]","",l.upper()))
        if cur and len(seqs) < maxn: seqs.append("".join(cur).upper())
    except: pass
    return [s[:maxlen] for s in seqs if len(s) >= 20]

K3 = ["".join(p) for p in product("ATGC",repeat=3)]
K4 = ["".join(p) for p in product("ATGC",repeat=4)]

def to_vec(seq, kl):
    k = len(kl[0])
    cnt = Counter(seq[i:i+k] for i in range(len(seq)-k+1) if all(c in "ATGC" for c in seq[i:i+k]))
    tot = sum(cnt.values()) or 1
    return np.array([cnt.get(km,0)/tot for km in kl], dtype=np.float32)

def gc(seq):
    t = sum(seq.count(n) for n in "ATGC") or 1
    return (seq.count("G")+seq.count("C"))/t*100

# ── Load data ──────────────────────────────────────────────────────────
print("Loading sequences ..."); sys.stdout.flush()
SD = {}
comp_rows = []
for sp, cfg in SPECIES.items():
    h,i = [],[]
    for rel in cfg["h"]:
        p = DATA_DIR/rel
        if p.exists(): h.extend(parse_fasta(p))
    for rel in cfg["i"]:
        p = DATA_DIR/rel
        if p.exists(): i.extend(parse_fasta(p))
    SD[sp] = {"h":h,"i":i}
    print(f"  {cfg['s']:12s} H={len(h):4d} I={len(i):4d}"); sys.stdout.flush()
    for seq in h:
        t=sum(seq.count(n) for n in "ATGC") or 1
        comp_rows.append({"Sp":cfg["s"],"Lbl":"Healthy","GC":gc(seq),
            "A":seq.count("A")/t*100,"T":seq.count("T")/t*100,"G":seq.count("G")/t*100,"C":seq.count("C")/t*100,"Len":len(seq)})
    for seq in i:
        t=sum(seq.count(n) for n in "ATGC") or 1
        comp_rows.append({"Sp":cfg["s"],"Lbl":"Infected","GC":gc(seq),
            "A":seq.count("A")/t*100,"T":seq.count("T")/t*100,"G":seq.count("G")/t*100,"C":seq.count("C")/t*100,"Len":len(seq)})

df = pd.DataFrame(comp_rows)

# ── K-mer mean vectors ─────────────────────────────────────────────────
print("Computing k-mer vectors ..."); sys.stdout.flush()
v3h={sp: np.vstack([to_vec(s,K3) for s in SD[sp]["h"]]).mean(0) if SD[sp]["h"] else np.zeros(len(K3)) for sp in SP}
v3i={sp: np.vstack([to_vec(s,K3) for s in SD[sp]["i"]]).mean(0) if SD[sp]["i"] else np.zeros(len(K3)) for sp in SP}
v4h={sp: np.vstack([to_vec(s,K4) for s in SD[sp]["h"]]).mean(0) if SD[sp]["h"] else np.zeros(len(K4)) for sp in SP}

# ── Correlation matrices ───────────────────────────────────────────────
print("Computing correlations ..."); sys.stdout.flush()
PEAR=np.zeros((N,N)); SPEAR=np.zeros((N,N)); JSD=np.zeros((N,N))
for i,s1 in enumerate(SP):
    for j,s2 in enumerate(SP):
        a,b = v3h[s1], v3h[s2]
        if a.any() and b.any():
            PEAR[i,j],_  = pearsonr(a,b)
            SPEAR[i,j],_ = spearmanr(a,b)
            p=a/(a.sum()+1e-12); q=b/(b.sum()+1e-12)
            JSD[i,j] = float(jensenshannon(p,q))
        else:
            PEAR[i,j]=SPEAR[i,j]=JSD[i,j]=0.0

# ── PCA ────────────────────────────────────────────────────────────────
print("PCA ..."); sys.stdout.flush()
pca_vecs,pca_sp,pca_lbl=[],[],[]
for sp in SP:
    for seq in SD[sp]["h"][:40]: pca_vecs.append(to_vec(seq,K3)); pca_sp.append(sp); pca_lbl.append("Healthy")
    for seq in SD[sp]["i"][:40]: pca_vecs.append(to_vec(seq,K3)); pca_sp.append(sp); pca_lbl.append("Infected")
Xp=StandardScaler().fit_transform(np.array(pca_vecs))
pca=PCA(n_components=2,random_state=42); Xpc=pca.fit_transform(Xp)

# ── ANOVA on GC ────────────────────────────────────────────────────────
gc_grp={SPECIES[sp]["s"]: df[df["Sp"]==SPECIES[sp]["s"]]["GC"].values for sp in SP}
valid_grps = [g for g in gc_grp.values() if len(g)>1]
if len(valid_grps)>=2:
    F,pv = stats.f_oneway(*valid_grps)
else:
    F,pv = 0.0, 1.0

# ── influenza k-mer enrichment ─────────────────────────────────────────
inf_sp = [sp for sp in SP if v3i[sp].any()]
enrich=[]
for sp in inf_sp:
    for km in INFLUENZA_KMERS:
        idx=K3.index(km)
        h_f=v3h[sp][idx]; i_f=v3i[sp][idx]
        enrich.append({"Sp":SPECIES[sp]["s"],"km":km,"H":h_f*100,"I":i_f*100,"FC":i_f/(h_f+1e-9),"Fn":INFLUENZA_KMERS[km]})
edf=pd.DataFrame(enrich)

print("Generating plots ..."); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 1 — Correlation + JSD
# ════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(22,7))
fig.suptitle("Inter-Species 3-mer Genomic Similarity Analysis",fontsize=16,fontweight="bold",color="#f8fafc")

for ax,mat,title,cmap,vmin,vmax in [
    (axes[0],PEAR,"Pearson Correlation","YlOrRd",0.8,1.0),
    (axes[1],SPEAR,"Spearman Correlation","PuBuGn",0.8,1.0),
    (axes[2],JSD,"Jensen-Shannon Divergence",LinearSegmentedColormap.from_list("js",["#10b981","#f59e0b","#f43f5e"]),0,0.3)]:
    im=ax.imshow(mat,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels(SHORTS,rotation=30,ha="right",fontsize=10)
    ax.set_yticks(range(N)); ax.set_yticklabels(SHORTS,fontsize=10)
    ax.set_title(title,fontsize=12)
    for i in range(N):
        for j in range(N):
            ax.text(j,i,f"{mat[i,j]:.3f}",ha="center",va="center",fontsize=9,fontweight="bold",
                    color="black" if (title!="Jensen-Shannon Divergence" and mat[i,j]>0.9) or (title=="Jensen-Shannon Divergence" and mat[i,j]<0.1) else "white")
    fig.colorbar(im,ax=ax,pad=0.02)

fig.tight_layout()
fig.savefig(OUT_DIR/"01_correlation_jsd.png",dpi=150,bbox_inches="tight",facecolor=DARK)
plt.close(fig)
print("  Saved 01_correlation_jsd.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 2 — Dendrogram
# ════════════════════════════════════════════════════════
dv,dl=[],[]
for sp in SP:
    dv.append(v3h[sp]); dl.append(f"{SPECIES[sp]['s']}\n(Healthy)")
    if v3i[sp].any(): dv.append(v3i[sp]); dl.append(f"{SPECIES[sp]['s']}\n(Infected)")
Z=linkage(np.array(dv),method="ward")
fig,ax=plt.subplots(figsize=(14,5)); fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
fig.suptitle("Hierarchical Clustering — Ward Linkage on 3-mer K-mer Space",fontsize=14,fontweight="bold",color="#f8fafc")
dendrogram(Z,labels=dl,ax=ax,leaf_font_size=9,color_threshold=0.5*max(Z[:,2]),above_threshold_color="#94a3b8")
ax.set_ylabel("Euclidean Distance (k-mer space)"); ax.tick_params(axis="x",colors="#e2e8f0")
fig.tight_layout()
fig.savefig(OUT_DIR/"02_dendrogram.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
print("  Saved 02_dendrogram.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 3 — PCA
# ════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(18,7))
fig.suptitle(f"PCA of 3-mer K-mer Space  —  PC1={pca.explained_variance_ratio_[0]*100:.1f}%  PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
             fontsize=14,fontweight="bold",color="#f8fafc")
ax=axes[0]
for idx,sp in enumerate(SP):
    mask=[i for i,s in enumerate(pca_sp) if s==sp]
    if mask: ax.scatter(Xpc[mask,0],Xpc[mask,1],color=SPECIES[sp]["c"],label=SPECIES[sp]["s"],alpha=0.7,s=35,edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Colored by Species"); ax.legend(fontsize=9,framealpha=0.3)
ax=axes[1]
for lbl,col,mk in [("Healthy",C2,"o"),("Infected",C4,"^")]:
    mask=[i for i,l in enumerate(pca_lbl) if l==lbl]
    if mask: ax.scatter(Xpc[mask,0],Xpc[mask,1],color=col,label=lbl,alpha=0.7,s=40,marker=mk,edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Colored by Health Status"); ax.legend(fontsize=11,framealpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/"03_pca_scatter.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
print("  Saved 03_pca_scatter.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 4 — GC + Nucleotide Composition
# ════════════════════════════════════════════════════════
fig,axes=plt.subplots(2,2,figsize=(18,12))
fig.suptitle("Nucleotide Composition & GC Content Analysis",fontsize=15,fontweight="bold",color="#f8fafc")
order=[SPECIES[s]["s"] for s in SP]

ax=axes[0,0]
if not df.empty:
    sns.boxplot(data=df,x="Sp",y="GC",hue="Lbl",palette={"Healthy":C2,"Infected":C4},ax=ax,order=order,width=0.6)
    ax.set_title("GC Content Distribution by Species"); ax.set_xlabel(""); ax.set_ylabel("GC (%)")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20,ha="right"); ax.legend(fontsize=10,framealpha=0.3)

ax=axes[0,1]
nt_means=np.array([[df[df["Sp"]==SPECIES[sp]["s"]][n].mean() for n in ["A","T","G","C"]] if not df[df["Sp"]==SPECIES[sp]["s"]].empty else [0,0,0,0] for sp in SP])
x=np.arange(N); w=0.18
for ni,(nt,col) in enumerate(zip(["A","T","G","C"],["#22d3ee","#a78bfa","#34d399","#fb923c"])):
    ax.bar(x+(ni-1.5)*w,nt_means[:,ni],w,label=nt,color=col,alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(order,rotation=20,ha="right")
ax.set_ylabel("Mean Frequency (%)"); ax.set_title("Nucleotide Composition (Healthy)"); ax.legend(fontsize=10,framealpha=0.3)

ax=axes[1,0]
if not df.empty:
    dflog=df.copy(); dflog["Len"]=np.log10(dflog["Len"].clip(lower=1))
    sns.violinplot(data=dflog,x="Sp",y="Len",hue="Lbl",palette={"Healthy":C2,"Infected":C4},ax=ax,order=order,inner="quartile")
    ax.set_title("Sequence Length Distribution (log10)"); ax.set_xlabel(""); ax.set_ylabel("log10(Length)")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=20,ha="right"); ax.legend(fontsize=10,framealpha=0.3)

ax=axes[1,1]
means=[df[df["Sp"]==SPECIES[sp]["s"]]["GC"].mean() if not df[df["Sp"]==SPECIES[sp]["s"]].empty else 0 for sp in SP]
stds =[df[df["Sp"]==SPECIES[sp]["s"]]["GC"].std()  if not df[df["Sp"]==SPECIES[sp]["s"]].empty else 0 for sp in SP]
ax.bar(range(N),means,yerr=stds,color=COLORS,capsize=5,error_kw={"ecolor":"#94a3b8","elinewidth":1.5},edgecolor="none",alpha=0.9)
for i,(m,s) in enumerate(zip(means,stds)): ax.text(i,m+s+0.5,f"{m:.1f}%",ha="center",fontsize=9,color="#e2e8f0")
ax.set_xticks(range(N)); ax.set_xticklabels(order,rotation=20,ha="right")
ax.set_ylabel("Mean GC (%)"); ax.set_title(f"Mean GC% per Species (ANOVA F={F:.2f}, p={pv:.2e})")
ax.axhline(np.nanmean(means),color=C3,lw=1.5,ls="--",label=f"Grand mean {np.nanmean(means):.1f}%"); ax.legend(fontsize=9,framealpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR/"04_gc_composition.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
print("  Saved 04_gc_composition.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 5 — Shared k-mer + Influenza-associated
# ════════════════════════════════════════════════════════
all_h_mean={km: np.mean([v3h[sp][K3.index(km)] for sp in SP if v3h[sp].any()]) for km in K3}
top20=sorted(all_h_mean,key=all_h_mean.get,reverse=True)[:20]

fig,axes=plt.subplots(1,2,figsize=(20,8))
fig.suptitle("Shared K-mer Profiles & Influenza-Associated K-mers",fontsize=15,fontweight="bold",color="#f8fafc")
ax=axes[0]
vals=[all_h_mean[km]*100 for km in top20]
cols=[C4 if km in INFLUENZA_KMERS else C1 for km in top20]
ax.barh(top20[::-1],vals[::-1],color=cols[::-1],alpha=0.9)
ax.set_xlabel("Mean Frequency (%) across all species"); ax.set_title("Top 20 Most Abundant 3-mers (Red=Influenza-Associated)")
ax.axvline(np.mean(vals),color=C3,lw=1.5,ls="--",label=f"Mean: {np.mean(vals):.2f}%"); ax.legend(fontsize=10,framealpha=0.3)

ax=axes[1]
inf_km_list=list(INFLUENZA_KMERS.keys())
x=np.arange(len(inf_km_list)); w=0.13
for idx,sp in enumerate(SP):
    vals2=[v3h[sp][K3.index(km)]*100 for km in inf_km_list]
    ax.bar(x+(idx-2.5)*w,vals2,w,label=SPECIES[sp]["s"],color=SPECIES[sp]["c"],alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(inf_km_list,rotation=30,ha="right",fontsize=10)
ax.set_ylabel("Frequency (%) in Healthy"); ax.set_title("Influenza K-mer Frequency — All 6 Species"); ax.legend(fontsize=8,framealpha=0.3,ncol=2)

fig.tight_layout()
fig.savefig(OUT_DIR/"05_kmer_susceptibility.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
print("  Saved 05_kmer_susceptibility.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 6 — K-mer shift infected vs healthy
# ════════════════════════════════════════════════════════
if inf_sp:
    top_shift=sorted(K3,key=lambda km: abs(np.mean([v3i[sp][K3.index(km)]-v3h[sp][K3.index(km)] for sp in inf_sp])),reverse=True)[:15]
    fig,ax=plt.subplots(figsize=(14,6)); fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
    fig.suptitle("3-mer Shift: Infected − Healthy (Top 15 Discriminative K-mers)",fontsize=14,fontweight="bold",color="#f8fafc")
    x=np.arange(len(top_shift)); w=0.22
    for idx,sp in enumerate(inf_sp):
        shifts=[(v3i[sp][K3.index(km)]-v3h[sp][K3.index(km)])*100 for km in top_shift]
        ax.bar(x+(idx-len(inf_sp)/2+0.5)*w,shifts,w,label=SPECIES[sp]["s"],color=SPECIES[sp]["c"],alpha=0.85)
    ax.axhline(0,color="#94a3b8",lw=1.2,ls="--"); ax.set_xticks(x); ax.set_xticklabels(top_shift,rotation=30,ha="right",fontsize=11)
    ax.set_ylabel("Frequency Shift (%) — Infected minus Healthy"); ax.legend(fontsize=10,framealpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"06_kmer_shift.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
    print("  Saved 06_kmer_shift.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 7 — 4-mer heatmap
# ════════════════════════════════════════════════════════
top4=sorted(K4,key=lambda km: np.std([v4h[sp][K4.index(km)] for sp in SP]),reverse=True)[:40]
mat4=np.array([[v4h[sp][K4.index(km)]*100 for km in top4] for sp in SP])
fig,ax=plt.subplots(figsize=(20,5)); fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
fig.suptitle("Top 40 Most Variable 4-mer Profiles Across Species (Healthy)",fontsize=14,fontweight="bold",color="#f8fafc")
im=ax.imshow(mat4,cmap="viridis",aspect="auto")
ax.set_yticks(range(N)); ax.set_yticklabels(SHORTS)
ax.set_xticks(range(len(top4))); ax.set_xticklabels(top4,rotation=45,ha="right",fontsize=8)
fig.colorbar(im,ax=ax,label="Frequency (%)",pad=0.02)
fig.tight_layout()
fig.savefig(OUT_DIR/"07_4mer_heatmap.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
print("  Saved 07_4mer_heatmap.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 8 — Influenza enrichment fold-change
# ════════════════════════════════════════════════════════
if not edf.empty:
    fig,axes=plt.subplots(1,2,figsize=(18,8))
    fig.suptitle("Influenza K-mer Enrichment: Infected vs Healthy",fontsize=14,fontweight="bold",color="#f8fafc")
    ax=axes[0]
    try:
        pivot=edf.pivot(index="km",columns="Sp",values="FC")
        cmap_fc=LinearSegmentedColormap.from_list("fc",["#3b82f6","#f8fafc","#f43f5e"])
        im=ax.imshow(pivot.values,cmap=cmap_fc,vmin=0.5,vmax=2.0,aspect="auto")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,rotation=20,ha="right",fontsize=10)
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index,fontsize=11)
        for ii in range(len(pivot.index)):
            for jj in range(len(pivot.columns)):
                v=pivot.values[ii,jj]
                ax.text(jj,ii,f"{v:.2f}x",ha="center",va="center",fontsize=9,fontweight="bold",color="black" if 0.8<v<1.2 else "white")
        fig.colorbar(im,ax=ax,label="Fold Change",pad=0.02)
    except Exception as e:
        ax.text(0.5,0.5,str(e),transform=ax.transAxes,ha="center",color="#f87171")
    ax.set_title("Fold Change Heatmap (Infected/Healthy)")
    ax=axes[1]
    for sp in inf_sp:
        sub=edf[edf["Sp"]==SPECIES[sp]["s"]]
        ax.scatter(sub["H"],sub["I"],color=SPECIES[sp]["c"],s=80,alpha=0.8,label=SPECIES[sp]["s"],edgecolors="none")
        for _,row in sub.iterrows():
            if row["km"] in ["ATG","GGC","CAA"]: ax.annotate(row["km"],(row["H"],row["I"]),fontsize=8,color="#e2e8f0",xytext=(3,3),textcoords="offset points")
    mx=max(edf[["H","I"]].max().max(),0.01)
    ax.plot([0,mx],[0,mx],"--",color="#94a3b8",lw=1.2,label="No change")
    ax.set_xlabel("Healthy Freq (%)"); ax.set_ylabel("Infected Freq (%)")
    ax.set_title("Healthy vs Infected Frequency of Influenza K-mers"); ax.legend(fontsize=9,framealpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"08_influenza_enrichment.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
    print("  Saved 08_influenza_enrichment.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# FIGURE 9 — Pairwise t-test on GC
# ════════════════════════════════════════════════════════
sl=SHORTS; nn=len(sl)
pmat=np.ones((nn,nn))
for i,s1 in enumerate(sl):
    for j,s2 in enumerate(sl):
        g1=gc_grp.get(s1,np.array([])); g2=gc_grp.get(s2,np.array([]))
        if i!=j and len(g1)>1 and len(g2)>1:
            _,pv2=stats.ttest_ind(g1,g2,equal_var=False); pmat[i,j]=pv2
fig,ax=plt.subplots(figsize=(9,7)); fig.patch.set_facecolor(DARK); ax.set_facecolor(CARD)
fig.suptitle(f"Pairwise Welch's t-test on GC%  |  ANOVA F={F:.2f}, p={pv:.2e}",fontsize=13,fontweight="bold",color="#f8fafc")
nlp=-np.log10(pmat+1e-300)
im=ax.imshow(nlp,cmap="hot",aspect="auto")
ax.set_xticks(range(nn)); ax.set_xticklabels(sl,rotation=30,ha="right",fontsize=10)
ax.set_yticks(range(nn)); ax.set_yticklabels(sl,fontsize=10)
for i in range(nn):
    for j in range(nn):
        sig="***" if pmat[i,j]<0.001 else ("**" if pmat[i,j]<0.01 else ("*" if pmat[i,j]<0.05 else "ns"))
        ax.text(j,i,sig,ha="center",va="center",fontsize=10,fontweight="bold",color="white" if nlp[i,j]>1 else "black")
fig.colorbar(im,ax=ax,label="-log10(p-value)",pad=0.02)
fig.tight_layout()
fig.savefig(OUT_DIR/"09_pairwise_ttest.png",dpi=150,bbox_inches="tight",facecolor=DARK); plt.close(fig)
print("  Saved 09_pairwise_ttest.png"); sys.stdout.flush()

# ════════════════════════════════════════════════════════
# Save JSON summary
# ════════════════════════════════════════════════════════
common_inf_kmers=[km for km in INFLUENZA_KMERS if all(v3h[sp][K3.index(km)]>0 for sp in SP)]
summary={
    "species_sequence_counts":{sp:{"healthy":len(SD[sp]["h"]),"infected":len(SD[sp]["i"])} for sp in SP},
    "anova_gc":{"F":float(F),"p":float(pv),"significant":bool(pv<0.05)},
    "pca_variance":{"PC1":float(pca.explained_variance_ratio_[0]),"PC2":float(pca.explained_variance_ratio_[1])},
    "pearson_correlation":{f"{SHORTS[i]}_vs_{SHORTS[j]}":float(PEAR[i,j]) for i in range(N) for j in range(i+1,N)},
    "js_divergence":{f"{SHORTS[i]}_vs_{SHORTS[j]}":float(JSD[i,j]) for i in range(N) for j in range(i+1,N)},
    "common_influenza_kmers_in_all_species":common_inf_kmers,
    "influenza_kmer_details":{km:{"function":fn,"mean_freq_healthy_pct":float(all_h_mean.get(km,0)*100),"in_all_species":km in common_inf_kmers} for km,fn in INFLUENZA_KMERS.items()},
    "mean_gc_per_species":{SPECIES[sp]["s"]:float(np.mean(gc_grp[SPECIES[sp]["s"]])) if len(gc_grp.get(SPECIES[sp]["s"],[])) else None for sp in SP}
}
with open(OUT_DIR/"genomic_analysis_summary.json","w") as f: json.dump(summary,f,indent=2)

# ════════════════════════════════════════════════════════
# Print final report
# ════════════════════════════════════════════════════════
print("\n"+"="*68)
print("  GENOMIC CORRELATION ANALYSIS — COMPLETE SUMMARY REPORT")
print("="*68)
print(f"\n{'Species':<25} {'Healthy':>8} {'Infected':>9} {'Mean GC%':>10} {'SD GC%':>8}")
print("-"*65)
for sp in SP:
    s=SPECIES[sp]["s"]; g=gc_grp.get(s,np.array([0]))
    print(f"  {sp:<23} {len(SD[sp]['h']):>8} {len(SD[sp]['i']):>9} {g.mean():>10.2f} {g.std():>8.2f}")

print(f"\nPCA Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"ANOVA on GC%: F={F:.4f}, p={pv:.4e} {'(SIGNIFICANT ✓)' if pv<0.05 else ''}")
print(f"\nPearson Correlation between any two species:")
for i,s1 in enumerate(SP):
    for j,s2 in enumerate(SP):
        if i<j: print(f"  {SHORTS[i]:12s} vs {SHORTS[j]:12s}: Pearson={PEAR[i,j]:.4f}  Spearman={SPEAR[i,j]:.4f}  JSD={JSD[i,j]:.4f}")
print(f"\nInfluenza K-mers present in ALL 6 healthy species:")
for km in common_inf_kmers:
    print(f"  {km}  — {INFLUENZA_KMERS[km]}  (mean: {all_h_mean.get(km,0)*100:.3f}%)")
print(f"\nAll 9 plots + JSON summary saved to: {OUT_DIR}")
print("="*68)
