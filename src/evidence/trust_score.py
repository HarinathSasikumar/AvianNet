"""
trust_score.py
Calculates the Overall Trust Score by combining:
  - Model Confidence  (from the ML pipeline prediction)
  - Scientific Evidence Score (from evidence_score.py)

Weights are configurable via score_utils.SCORING_CONFIG.
Does NOT modify any ML model or prediction logic.
"""

import logging
from typing import Dict, Any
from .score_utils import SCORING_CONFIG, clamp, validate_config
from .paper_analyzer import analyze_papers
from .evidence_score import calculate_evidence_score, generate_evidence_explanation

logger = logging.getLogger(__name__)


def calculate_trust_score(model_confidence: float, evidence_score: float) -> float:
    """
    Calculate the Overall Trust Score as a weighted combination of
    model confidence and scientific evidence.

    Formula:
        trust = (confidence * confidence_weight)
              + (evidence_score * evidence_weight)

    Args:
        model_confidence: float in [0.0, 1.0] from the ML model.
        evidence_score:   float in [0.0, 100.0] from calculate_evidence_score().

    Returns:
        Float percentage in [0.0, 100.0].
    """
    validate_config(SCORING_CONFIG)

    confidence_pct = clamp(model_confidence * 100.0, 0.0, 100.0)
    evidence_pct   = clamp(evidence_score,            0.0, 100.0)

    trust = (
        confidence_pct * SCORING_CONFIG["confidence_weight"]
        + evidence_pct * SCORING_CONFIG["evidence_weight"]
    )

    trust_clamped = clamp(trust, 0.0, 100.0)
    logger.info(
        f"Trust Score — confidence: {confidence_pct:.2f}%, evidence: {evidence_pct:.2f}%, "
        f"trust: {trust_clamped:.2f}%"
    )
    return round(trust_clamped, 2)


def generate_full_evidence_report(prediction_data: Dict[str, Any], papers: list) -> Dict[str, Any]:
    """
    End-to-end function that builds the complete Evidence & Trust report.

    Args:
        prediction_data: dict containing:
            - "prediction"        (str)
            - "confidence"        (float 0–1)
            - "mutation_gene"     (str)
            - "mutation_positions"(str)
        papers: list of paper dicts retrieved from the RAG module.

    Returns:
        Complete evidence report dict — safe to serialize as JSON.
    """
    prediction  = prediction_data.get("prediction",         "Unknown")
    confidence  = float(prediction_data.get("confidence",   0.0))
    mutation    = prediction_data.get("mutation_gene",       "Unknown Gene")
    position    = prediction_data.get("mutation_positions",  "Unknown")

    # 1. Classify papers
    annotated_papers, counts = analyze_papers(papers)

    # 2. Evidence Score
    evidence_score = calculate_evidence_score(counts)

    # 3. Trust Score
    trust_score = calculate_trust_score(confidence, evidence_score)

    # 4. Explanation
    explanation = generate_evidence_explanation(
        evidence_score, counts, mutation, prediction
    )

    report = {
        "prediction":             prediction,
        "model_confidence_pct":   round(confidence * 100, 2),
        "mutation":               mutation,
        "position":               position,
        "papers_analyzed":        len(papers),
        "supporting_papers":      counts["supporting"],
        "neutral_papers":         counts["neutral"],
        "contradicting_papers":   counts["contradicting"],
        "scientific_evidence_score": evidence_score,
        "overall_trust_score":    trust_score,
        "ai_explanation":         explanation,
        "annotated_papers":       annotated_papers,
        "scoring_weights": {
            "confidence_weight": SCORING_CONFIG["confidence_weight"],
            "evidence_weight":   SCORING_CONFIG["evidence_weight"],
        },
    }

    return report
