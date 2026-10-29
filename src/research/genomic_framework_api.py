"""
src/research/genomic_framework_api.py
======================================
FastAPI Router — Genomic Explainability, Robustness & Scientific Evidence Framework
Serves all outputs from Modules 1–5 to the dashboard.

Routes:
  GET /genomic/status
  GET /genomic/xai-consensus
  GET /genomic/perturbation
  GET /genomic/stability
  GET /genomic/fidelity
  GET /genomic/evidence
  GET /genomic/trust-report
  GET /genomic/species-summary
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/genomic", tags=["Genomic Framework"])

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
RESEARCH_DIR  = PROJECT_ROOT / "outputs" / "research"
XAI_DIR       = RESEARCH_DIR / "xai"
PERTURB_DIR   = RESEARCH_DIR / "perturbation"
EVIDENCE_DIR  = RESEARCH_DIR / "evidence"


def _load_json(path: Path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load {path}: {e}")
    return None


@router.get("/status")
async def framework_status():
    """Check which genomic framework modules have generated outputs."""
    files = {
        "xai_consensus":  (XAI_DIR / "xai_consensus.json").exists(),
        "perturbation":   (PERTURB_DIR / "perturbation_summary.json").exists(),
        "stability":      (XAI_DIR / "explanation_stability.json").exists(),
        "fidelity":       (XAI_DIR / "xai_fidelity.json").exists(),
        "evidence":       (EVIDENCE_DIR / "evidence_backed_explanations.json").exists(),
        "trust_report":   (RESEARCH_DIR / "genomic_trust_report.json").exists(),
    }
    return {"framework_ready": any(files.values()), "modules": files}


@router.get("/xai-consensus")
async def get_xai_consensus():
    data = _load_json(XAI_DIR / "xai_consensus.json")
    if data:
        return data
    raise HTTPException(status_code=404, detail="XAI consensus data not found. Run framework first.")


@router.get("/perturbation")
async def get_perturbation_summary():
    data = _load_json(PERTURB_DIR / "perturbation_summary.json")
    if data:
        return data
    raise HTTPException(status_code=404, detail="Perturbation data not found. Run framework first.")


@router.get("/stability")
async def get_stability_summary():
    data = _load_json(XAI_DIR / "explanation_stability.json")
    if data:
        return data
    raise HTTPException(status_code=404, detail="Stability data not found. Run framework first.")


@router.get("/fidelity")
async def get_fidelity_summary():
    data = _load_json(XAI_DIR / "xai_fidelity.json")
    if data:
        return data
    raise HTTPException(status_code=404, detail="Fidelity data not found. Run framework first.")


@router.get("/evidence")
async def get_evidence_summary():
    data = _load_json(EVIDENCE_DIR / "evidence_backed_explanations.json")
    if data:
        return data
    raise HTTPException(status_code=404, detail="Evidence data not found. Run framework first.")


@router.get("/trust-report")
async def get_trust_report():
    data = _load_json(RESEARCH_DIR / "genomic_trust_report.json")
    if data:
        return data
    raise HTTPException(status_code=404, detail="Trust report not found. Run framework first.")


@router.get("/species-summary")
async def get_species_summary():
    data = _load_json(RESEARCH_DIR / "genomic_trust_report.json")
    if data and "species_matrix" in data:
        return {"species_matrix": data["species_matrix"]}
    raise HTTPException(status_code=404, detail="Species summary not found.")
