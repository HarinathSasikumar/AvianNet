"""
evidence_score.py
Calculates a Scientific Evidence Score (0–100) based on the distribution
of Supporting, Neutral, and Contradicting papers retrieved via the RAG module.
Does NOT modify any ML model or prediction logic.
"""

import logging
from typing import Dict
from .score_utils import SCORING_CONFIG, clamp

logger = logging.getLogger(__name__)


def calculate_evidence_score(counts: Dict[str, int]) -> float:
    """
    Calculate a Scientific Evidence Score on a 0–100 scale.

    Formula (configurable in score_utils.py):
        raw = (n_supporting * SUPPORTING_SCORE)
            + (n_neutral    * NEUTRAL_SCORE)
            + (n_contradicting * CONTRADICTING_SCORE)

    The result is clamped to [0, 100].

    Args:
        counts: dict with keys 'supporting', 'neutral', 'contradicting' (int values).

    Returns:
        Float in [0.0, 100.0].
    """
    n_supporting    = counts.get("supporting",    0)
    n_neutral       = counts.get("neutral",       0)
    n_contradicting = counts.get("contradicting", 0)

    raw_score = (
        n_supporting    * SCORING_CONFIG["supporting_paper_score"]
        + n_neutral     * SCORING_CONFIG["neutral_paper_score"]
        + n_contradicting * SCORING_CONFIG["contradicting_paper_score"]
    )

    evidence_score = clamp(
        raw_score,
        SCORING_CONFIG["min_evidence_score"],
        SCORING_CONFIG["max_evidence_score"],
    )

    logger.info(
        f"Evidence Score — raw: {raw_score:.2f}, clamped: {evidence_score:.2f} "
        f"(S={n_supporting}, N={n_neutral}, C={n_contradicting})"
    )
    return round(evidence_score, 2)


def generate_evidence_explanation(
    evidence_score: float,
    counts: Dict[str, int],
    mutation: str,
    prediction: str,
) -> str:
    """
    Generate a human-readable explanation for the Evidence Score.

    Args:
        evidence_score: calculated score (0–100).
        counts:         paper classification counts.
        mutation:       detected mutation/gene label.
        prediction:     model prediction string ("Infected" / "Healthy").

    Returns:
        Explanation string.
    """
    n_s = counts.get("supporting",    0)
    n_n = counts.get("neutral",       0)
    n_c = counts.get("contradicting", 0)
    total = n_s + n_n + n_c

    if total == 0:
        return (
            "No PubMed papers were retrieved for this mutation context. "
            "The Evidence Score is unavailable and defaults to 0. "
            "Manual literature review is recommended."
        )

    if evidence_score >= 80:
        strength = "strong"
        qualifier = "well-supported by"
    elif evidence_score >= 50:
        strength = "moderate"
        qualifier = "partially supported by"
    elif evidence_score >= 20:
        strength = "weak"
        qualifier = "weakly supported by"
    else:
        strength = "very low"
        qualifier = "largely contradicted in"

    explanation = (
        f"The prediction '{prediction}' for the detected mutation in the {mutation} "
        f"shows {strength} scientific evidence (score: {evidence_score:.0f}/100). "
        f"Out of {total} retrieved PubMed paper(s), {n_s} are supporting, "
        f"{n_n} are neutral, and {n_c} are contradicting. "
        f"The mutation context is {qualifier} the current scientific literature."
    )

    if n_c > n_s:
        explanation += (
            " ⚠️ Contradicting evidence exceeds supporting evidence — "
            "additional experimental validation is strongly advised."
        )

    return explanation
