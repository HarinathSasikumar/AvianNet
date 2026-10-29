"""
score_utils.py
Configurable weights and shared utility functions for the Evidence & Trust scoring system.
All weights can be tuned here without touching any other module.
"""

import logging

logger = logging.getLogger(__name__)

# ─── Configurable Scoring Weights ───────────────────────────────────────────
SCORING_CONFIG = {
    # Weight of model confidence in the final Overall Trust Score (0.0 – 1.0)
    "confidence_weight": 0.55,

    # Weight of scientific evidence score in the final Overall Trust Score (0.0 – 1.0)
    "evidence_weight": 0.45,

    # Per-paper contribution to Evidence Score (0 – 100 scale)
    "supporting_paper_score":    10.0,   # positive signal
    "neutral_paper_score":        3.0,   # mild positive signal
    "contradicting_paper_score": -8.0,   # negative signal

    # Maximum clamp for Evidence Score
    "max_evidence_score": 100.0,
    "min_evidence_score":   0.0,

    # Keyword lists used by PaperAnalyzer
    "supporting_keywords": [
        "pathogenic", "virulent", "virulence", "increased transmissibility",
        "high pathogenicity", "h5n1", "h7n9", "h9n2", "avian influenza",
        "hemagglutinin mutation", "receptor binding", "adaptation",
        "pandemic risk", "zoonotic", "interspecies",
    ],
    "contradicting_keywords": [
        "no evidence", "not associated", "low pathogenicity",
        "benign mutation", "protective", "attenuated",
        "no significant", "not detected", "absence of",
    ],
}


def validate_config(config: dict) -> bool:
    """Ensure confidence_weight + evidence_weight sum to 1.0 (±0.01 tolerance)."""
    total = config["confidence_weight"] + config["evidence_weight"]
    if abs(total - 1.0) > 0.01:
        logger.warning(
            f"Scoring weights do not sum to 1.0 (got {total:.3f}). "
            "Results may be inaccurate. Adjust SCORING_CONFIG in score_utils.py."
        )
        return False
    return True


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))
