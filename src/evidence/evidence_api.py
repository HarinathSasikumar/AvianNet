"""
evidence_api.py
FastAPI router for the Evidence & Trust Score module.
Exposes a single new endpoint: POST /evidence/report
Does NOT modify any existing RAG or ML endpoint.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.rag.rag_pipeline import RAGPipeline
from src.evidence.trust_score import generate_full_evidence_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evidence", tags=["Evidence & Trust Score"])

# Shared RAG pipeline instance (stateless retrieval only)
_rag = RAGPipeline()


class EvidenceRequest(BaseModel):
    prediction:        str   = Field(..., example="Infected")
    confidence:        float = Field(..., ge=0.0, le=1.0, example=0.986)
    mutation_gene:     str   = Field(..., example="HA Gene")
    mutation_positions: str  = Field(..., example="125-138")
    papers:            Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Pre-fetched paper list. If omitted the module will query RAG automatically."
    )


class EvidenceResponse(BaseModel):
    prediction:                str
    model_confidence_pct:      float
    mutation:                  str
    position:                  str
    papers_analyzed:           int
    supporting_papers:         int
    neutral_papers:            int
    contradicting_papers:      int
    scientific_evidence_score: float
    overall_trust_score:       float
    ai_explanation:            str
    annotated_papers:          List[Dict[str, Any]]
    scoring_weights:           Dict[str, float]


@router.post("/report", response_model=EvidenceResponse, summary="Generate Evidence & Trust Score Report")
async def evidence_report(request: EvidenceRequest):
    """
    Generate a full Evidence & Trust Score report for a completed prediction.

    - If `papers` are supplied in the request body, they are used directly.
    - Otherwise the module automatically fetches papers from PubMed via the RAG pipeline.
    - Model confidence and scientific evidence are combined into an Overall Trust Score.
    - No ML models are retrained or modified.
    """
    try:
        # Step 1: Obtain papers
        if request.papers:
            papers = request.papers
            logger.info(f"Using {len(papers)} pre-supplied papers for evidence analysis.")
        else:
            query = f"Avian influenza {request.mutation_gene} mutation"
            logger.info(f"Fetching papers from RAG pipeline for query: '{query}'")
            papers = _rag.retriever.run_retrieval(query, top_k=10)

        # Step 2: Generate full report
        report = generate_full_evidence_report(request.model_dump(), papers)
        return report

    except Exception as e:
        logger.error(f"Evidence report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evidence report generation failed: {str(e)}")
