"""
src/research/evidence_explanation.py
======================================
Module 5 — Evidence-Backed Explanation
ADDITIVE MODULE: Does not modify any existing code.

Integrates XAI findings with live PubMed searches via the existing RAG pipeline
to generate evidence-backed scientific summaries for each sequence prediction.

Outputs:
  outputs/research/evidence/evidence_backed_explanations.csv
  outputs/research/evidence/evidence_backed_explanations.json

IMPORTANT: PubMed evidence is retrieved to provide scientific context ONLY.
It does NOT prove biological causality. Each record clearly distinguishes:
  - AI model finding
  - XAI computational finding
  - Perturbation analysis finding
  - Scientific literature evidence
"""

import json
import csv
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR  = PROJECT_ROOT / "outputs" / "research" / "evidence"
XAI_DIR       = PROJECT_ROOT / "outputs" / "research" / "xai"
REPORTS_DIR   = PROJECT_ROOT / "outputs" / "research" / "reports"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SPECIES_METADATA = {
    "GALLUS":  {"scientific": "Gallus gallus",       "common": "Chicken"},
    "DUCK":    {"scientific": "Anas platyrhynchos",  "common": "Mallard Duck"},
    "TURKEY":  {"scientific": "Meleagris gallopavo", "common": "Wild Turkey"},
    "PIGEON":  {"scientific": "Columba livia",        "common": "Rock Pigeon"},
    "AVIAN":   {"scientific": "Avian spp.",           "common": "Avian (mixed)"},
}

AVIAN_INFLUENZA_STRAINS = ["H5N1", "H5N2", "H5N6", "H5N8", "H7N3", "H7N7", "H7N9", "H9N2"]


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


def _infer_strain(seq_id: str, label: str) -> Optional[str]:
    sid = seq_id.upper()
    for strain in AVIAN_INFLUENZA_STRAINS:
        if strain in sid:
            return strain
    if label == "Infected":
        return "H5N1"  # most common in dataset
    return None


# ---------------------------------------------------------------------------
# PubMed retrieval via existing RAG pipeline
# ---------------------------------------------------------------------------

def _search_pubmed_via_rag(query: str, top_k: int = 5) -> List[Dict]:
    """
    Uses the existing RAG pipeline to search PubMed.
    Returns a list of paper dicts with pmid, title, year, journal, relevance.
    """
    try:
        from src.rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        papers = pipeline.retriever.run_retrieval(query, top_k=top_k)
        if papers and isinstance(papers, list):
            return papers
    except Exception as e:
        logger.warning(f"RAG pipeline unavailable: {e}")

    # Fallback: try direct NCBI Entrez search
    return _search_pubmed_entrez(query, top_k)


def _search_pubmed_entrez(query: str, top_k: int = 5) -> List[Dict]:
    """Direct NCBI Entrez API fallback. Returns real PubMed records."""
    try:
        from Bio import Entrez
        Entrez.email = "avian.dna.research@example.com"

        # Search
        handle = Entrez.esearch(
            db="pubmed",
            term=query + " [Title/Abstract]",
            retmax=top_k,
            sort="relevance",
        )
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        if not ids:
            return []

        time.sleep(0.35)  # respect NCBI rate limit

        # Fetch summaries
        handle2 = Entrez.esummary(db="pubmed", id=",".join(ids), retmode="xml")
        summaries = Entrez.read(handle2)
        handle2.close()

        results = []
        for s in summaries:
            pmid  = str(s.get("Id", ""))
            title = str(s.get("Title", ""))
            year  = str(s.get("PubDate", ""))[:4]
            journal = str(s.get("Source", ""))
            results.append({
                "pmid":    pmid,
                "title":   title,
                "year":    year,
                "journal": journal,
                "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return results

    except ImportError:
        logger.warning("BioPython not installed. Cannot search PubMed directly.")
        return []
    except Exception as e:
        logger.warning(f"PubMed Entrez search failed: {e}")
        return []


def _build_search_query(
    prediction: str,
    species_tag: str,
    xai_region: str,
    strain: Optional[str],
) -> str:
    """
    Build a targeted PubMed query based on XAI and prediction results.
    Does NOT claim causality — searches for supporting/contextual literature.
    """
    species_info = SPECIES_METADATA.get(species_tag, {"scientific": "avian"})
    scientific   = species_info["scientific"]

    if prediction == "Infected" and strain:
        base = f"Avian influenza {strain} {scientific} genomic sequence classification"
    elif prediction == "Infected":
        base = f"Avian influenza {scientific} genomic classification machine learning"
    else:
        base = f"{scientific} healthy genomic sequence analysis influenza surveillance"

    if xai_region and xai_region != "N/A":
        base += " gene region mutation hotspot"

    return base


# ---------------------------------------------------------------------------
# Evidence record builder
# ---------------------------------------------------------------------------

def build_evidence_record(
    seq_id:      str,
    sequence:    str,
    label:       str,
    species_tag: str,
    seq_index:   int,
    top_k_papers: int = 5,
) -> Dict[str, Any]:
    """
    Build a full evidence-backed explanation record for a single sequence.
    """
    strain       = _infer_strain(seq_id, label)
    species_info = SPECIES_METADATA.get(species_tag, {"scientific": species_tag, "common": species_tag})

    # Load XAI consensus for this sequence
    xai_region   = "N/A"
    xai_consensus = 0.0
    consensus_data = _load_json(XAI_DIR / "xai_consensus.json")
    if consensus_data:
        for rec in consensus_data.get("per_sequence", []):
            if seq_id in rec.get("seq_id", ""):
                xai_region    = rec.get("attention_region", "N/A")
                xai_consensus = rec.get("consensus_score", 0.0)
                break

    # Load perturbation stats
    perturbation_sensitivity = 0.0
    perturbation_flip = False
    perturb_data = _load_json(
        PROJECT_ROOT / "outputs" / "research" / "perturbation" / "perturbation_summary.json"
    )
    if perturb_data:
        xai_stats = perturb_data.get("xai_region_stats", {})
        perturbation_sensitivity = xai_stats.get("mean_prob_diff", 0.0)
        perturbation_flip = xai_stats.get("flip_rate", 0.0) > 0.5

    # Load fidelity
    fidelity_score = 0.0
    fidelity_data = _load_json(XAI_DIR / "xai_fidelity.json")
    if fidelity_data:
        fidelity_score = fidelity_data.get("mean_fidelity_score", 0.0)

    # Build query and search PubMed
    query   = _build_search_query(label, species_tag, xai_region, strain)
    papers  = _search_pubmed_via_rag(query, top_k=top_k_papers)
    n_papers = len(papers)

    # Determine evidence status
    if n_papers >= 3:
        evidence_status = "Supporting literature identified"
        evidence_strength = "Moderate-to-Strong"
    elif n_papers >= 1:
        evidence_status = "Limited supporting literature identified"
        evidence_strength = "Weak"
    else:
        evidence_status = "No supporting literature retrieved (network or API unavailable)"
        evidence_strength = "None"

    # Build evidence summary
    summary_lines = [
        f"AI MODEL FINDING: Sequence classified as {label}.",
        f"XAI COMPUTATIONAL FINDING: Important DNA region identified at positions {xai_region}.",
        f"XAI CONSENSUS: {round(xai_consensus * 100, 1)}% agreement across SHAP, LIME, Attention.",
        f"PERTURBATION ANALYSIS: Mean confidence change = {round(perturbation_sensitivity, 4)} (model sensitivity).",
        f"XAI FIDELITY SCORE: {round(fidelity_score, 4)} (XAI regions vs random masking effect difference).",
        f"SCIENTIFIC LITERATURE: {n_papers} relevant PubMed paper(s) identified via search.",
        f"EVIDENCE STATUS: {evidence_status}.",
        "IMPORTANT CAVEAT: Supporting literature provides scientific context ONLY. "
        "It does NOT prove that identified DNA positions cause Avian Influenza. "
        "XAI results reflect model behavior, not biological ground truth.",
    ]

    return {
        "seq_id":                  seq_id,
        "species_tag":             species_tag,
        "scientific_name":         species_info["scientific"],
        "common_name":             species_info["common"],
        "strain":                  strain or "N/A",
        "prediction":              label,
        "xai_important_region":    xai_region,
        "xai_consensus_score":     round(xai_consensus, 4),
        "perturbation_sensitivity":round(perturbation_sensitivity, 4),
        "fidelity_score":          round(fidelity_score, 4),
        "pubmed_search_query":     query,
        "n_papers_retrieved":      n_papers,
        "evidence_status":         evidence_status,
        "evidence_strength":       evidence_strength,
        "evidence_summary":        " | ".join(summary_lines),
        "papers": [
            {
                "pmid":    p.get("pmid", p.get("id", "")),
                "title":   p.get("title", ""),
                "year":    p.get("year", p.get("pub_date", "")),
                "journal": p.get("journal", p.get("source", "")),
                "url":     p.get("url", ""),
            }
            for p in papers
        ],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evidence_explanation(
    species_filter: Optional[str] = None,
    n_sequences:    int = 15,
    top_k_papers:   int = 5,
) -> Dict[str, Any]:
    """
    Module 5 main entry point.
    """
    logger.info("=== Evidence-Backed Explanation START ===")
    timestamp = datetime.utcnow().isoformat()

    from src.research.perturbation_analysis import _build_test_sequences_from_data
    test_seqs = _build_test_sequences_from_data()

    if species_filter and species_filter.upper() != "ALL":
        test_seqs = [s for s in test_seqs if s["species_tag"] == species_filter.upper()]

    # Limit to fewer sequences to respect PubMed rate limits
    test_seqs = test_seqs[:n_sequences]
    records = []

    for i, seq_rec in enumerate(test_seqs):
        logger.info(f"  Evidence [{i+1}/{len(test_seqs)}]: {seq_rec['seq_id'][:40]}")
        rec = build_evidence_record(
            seq_id=seq_rec["seq_id"],
            sequence=seq_rec["sequence"],
            label=seq_rec["label"],
            species_tag=seq_rec["species_tag"],
            seq_index=i,
            top_k_papers=top_k_papers,
        )
        records.append(rec)
        time.sleep(0.4)  # Respect NCBI rate limit

    # Species-wise summary
    species_groups: Dict[str, List] = {}
    for r in records:
        species_groups.setdefault(r["species_tag"], []).append(r)

    species_summary = {}
    for tag, recs in species_groups.items():
        species_info = SPECIES_METADATA.get(tag, {"scientific": tag, "common": tag})
        species_summary[tag] = {
            "scientific_name":   species_info["scientific"],
            "n_sequences":       len(recs),
            "total_papers":      sum(r["n_papers_retrieved"] for r in recs),
            "mean_papers":       round(float(np.mean([r["n_papers_retrieved"] for r in recs])), 2),
            "evidence_status_counts": {
                status: sum(1 for r in recs if r["evidence_status"] == status)
                for status in set(r["evidence_status"] for r in recs)
            },
        }

    result = {
        "module":           "Evidence-Backed Explanation",
        "timestamp":        timestamp,
        "random_seed":      RANDOM_SEED,
        "n_sequences":      len(records),
        "total_papers":     sum(r["n_papers_retrieved"] for r in records),
        "species_summary":  species_summary,
        "per_sequence":     records,
        "caveats": [
            "PubMed searches provide scientific context for identified genomic patterns.",
            "Retrieved papers do NOT prove that specific nucleotides cause Avian Influenza.",
            "XAI findings are computational results from the trained AI model.",
            "Biological interpretation requires domain expert review.",
        ],
    }

    # ---- Save ----
    json_path = EVIDENCE_DIR / "evidence_backed_explanations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    csv_path = EVIDENCE_DIR / "evidence_backed_explanations.csv"
    if records:
        flat_records = []
        for r in records:
            flat = {k: v for k, v in r.items() if k != "papers"}
            flat["paper_pmids"] = "|".join(str(p.get("pmid", "")) for p in r.get("papers", []))
            flat["paper_titles"] = "|".join(str(p.get("title", "")) for p in r.get("papers", []))
            flat_records.append(flat)
        if flat_records:
            fieldnames = [
                "seq_id", "species_tag", "scientific_name", "common_name", "strain",
                "prediction", "xai_important_region", "xai_consensus_score",
                "perturbation_sensitivity", "fidelity_score",
                "pubmed_search_query", "n_papers_retrieved",
                "evidence_status", "evidence_strength", "evidence_summary",
                "paper_pmids", "paper_titles",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(flat_records)
            logger.info(f"Saved: {csv_path}")

    logger.info(f"Evidence-Backed Explanation complete. {len(records)} records generated.")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evidence_explanation()
