"""
Part 1: Load sequences, compute k-mer matrices, save intermediate data.
Fast run: ~60s
"""
import sys, json, re, warnings, pickle
from pathlib import Path
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "outputs" / "genomic_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = {
    "Gallus gallus":       {"short":"G.gallus",  "healthy":["gallus_gallus/healthy.fasta","healthy/general/gallus_gallus_cds.fasta","healthy/general/gallus_gallus_gene.fasta","healthy/general/gallus_exon.fasta","healthy/general/gallus_gallus_cds_expanded.fasta"],      "infected":["gallus_gallus/infected.fasta","infected/H5/H5N1_sequences.fasta","infected/H5/H5N2_sequences.fasta","infected/H5/H5N6_sequences.fasta","infected/H5/H5N8_sequences.fasta","infected/H7/H7N9_sequences.fasta","infected/H7/H7N3_sequences.fasta","infected/H7/H7N7_sequences.fasta"]},
    "Anas platyrhynchos":  {"short":"A.platyr.", "healthy":["anas_platyrhynchos/healthy.fasta","healthy/general/duck_anas_platyrhynchos.fasta"],                                                                                                                               "infected":["anas_platyrhynchos/infected.fasta"]},
    "Meleagris gallopavo": {"short":"M.gallop.", "healthy":["meleagris_gallopavo/healthy.fasta","healthy/general/turkey_meleagris_gallopavo.fasta"],                                                                                                                           "infected":[]},
    "Anser anser":         {"short":"A.anser",   "healthy":["anser_anser/healthy.fasta"],                                                                                                                                                                                      "infected":[]},
    "Columba livia":       {"short":"C.livia",   "healthy":["columba_livia/healthy.fasta","healthy/general/pigeon_columba_livia.fasta"],                                                                                                                                       "infected":["columba_livia/infected.fasta"]},
    "Passer domesticus":   {"short":"P.domest.", "healthy":["passer_domesticus/healthy.fasta"],                                                                                                                                                                                "infected":[]},
}

def parse_fasta(path):
    seqs, cur = [], []
    try:
        with open(path,"r",errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if cur: seqs.append("".join(cur).upper())
                    cur = []
                else:
                    cleaned = re.sub(r"[^ATGCN]","",line.upper())
                    if cleaned: cur.append(cleaned)
        if cur: seqs.append("".join(cur).upper())
    except: pass
    return [s for s in seqs if len(s) >= 20]

def kmer_freq(seq, k):
    counts = Counter(seq[i:i+k] for i in range(len(seq)-k+1) if all(c in "ATGC" for c in seq[i:i+k]))
    total = sum(counts.values()) or 1
    return {km:cnt/total for km,cnt in counts.items()}

def all_kmers(k):
    return ["".join(p) for p in product("ATGC",repeat=k)]

def seq_to_vec(seq, k, kmer_list):
    freq = kmer_freq(seq, k)
    return np.array([freq.get(km,0.0) for km in kmer_list], dtype=float)

def gc_content(seq):
    gc = seq.count("G")+seq.count("C")
    total = sum(seq.count(n) for n in "ATGC") or 1
    return gc/total*100

kmers3 = all_kmers(3)
kmers4 = all_kmers(4)

print("="*60)
print("PART 1: Loading sequences & computing k-mer matrices")
print("="*60)

species_data = {}
comp_rows    = []

for sp, cfg in SPECIES.items():
    h_seqs, i_seqs = [], []
    for rel in cfg["healthy"]:
        p = DATA_DIR/rel
        if p.exists(): h_seqs.extend(parse_fasta(p)[:200])
    for rel in cfg["infected"]:
        p = DATA_DIR/rel
        if p.exists(): i_seqs.extend(parse_fasta(p)[:200])
    species_data[sp] = {"healthy": h_seqs, "infected": i_seqs}
    print(f"  {cfg['short']:12s}  healthy={len(h_seqs):4d}  infected={len(i_seqs):4d}")

    for seq in h_seqs[:200]:
        total = sum(seq.count(n) for n in "ATGC") or 1
        comp_rows.append({"Species":cfg["short"],"Label":"Healthy","GC%":gc_content(seq),
            "A%":seq.count("A")/total*100,"T%":seq.count("T")/total*100,
            "G%":seq.count("G")/total*100,"C%":seq.count("C")/total*100,"Length":len(seq)})
    for seq in i_seqs[:200]:
        total = sum(seq.count(n) for n in "ATGC") or 1
        comp_rows.append({"Species":cfg["short"],"Label":"Infected","GC%":gc_content(seq),
            "A%":seq.count("A")/total*100,"T%":seq.count("T")/total*100,
            "G%":seq.count("G")/total*100,"C%":seq.count("C")/total*100,"Length":len(seq)})

# Compute mean k-mer vectors per species
print("\nComputing mean k-mer vectors ...")
sp_names = list(SPECIES.keys())
sp3h, sp3i, sp4h = {}, {}, {}

for sp, data in species_data.items():
    if data["healthy"]:
        vecs = np.array([seq_to_vec(s, 3, kmers3) for s in data["healthy"][:150]])
        sp3h[sp] = vecs.mean(axis=0)
        v4 = np.array([seq_to_vec(s, 4, kmers4) for s in data["healthy"][:100]])
        sp4h[sp] = v4.mean(axis=0)
    else:
        sp3h[sp] = np.zeros(len(kmers3))
        sp4h[sp] = np.zeros(len(kmers4))

    if data["infected"]:
        vecs = np.array([seq_to_vec(s, 3, kmers3) for s in data["infected"][:150]])
        sp3i[sp] = vecs.mean(axis=0)
    else:
        sp3i[sp] = np.zeros(len(kmers3))
    print(f"  {SPECIES[sp]['short']:12s} done")

# PCA data
print("\nBuilding PCA data ...")
all_vecs, all_sp, all_labels = [], [], []
for sp, data in species_data.items():
    for seq in data["healthy"][:60]:
        all_vecs.append(seq_to_vec(seq,3,kmers3)); all_sp.append(sp); all_labels.append("Healthy")
    for seq in data["infected"][:60]:
        all_vecs.append(seq_to_vec(seq,3,kmers3)); all_sp.append(sp); all_labels.append("Infected")

# Save everything
save_data = {
    "sp_names": sp_names,
    "sp3h":   {k:v.tolist() for k,v in sp3h.items()},
    "sp3i":   {k:v.tolist() for k,v in sp3i.items()},
    "sp4h":   {k:v.tolist() for k,v in sp4h.items()},
    "kmers3": kmers3,
    "kmers4": kmers4,
    "comp_rows": comp_rows,
    "pca_vecs":  np.array(all_vecs).tolist(),
    "pca_sp":    all_sp,
    "pca_labels": all_labels,
}
cache_path = OUT_DIR / "kmer_cache.json"
with open(cache_path,"w") as f:
    json.dump(save_data, f)

print(f"\nCache saved to: {cache_path}")
print("PART 1 COMPLETE")
