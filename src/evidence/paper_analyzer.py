"""
paper_analyzer.py
Classifies each retrieved PubMed paper as Supporting, Neutral, or Contradicting
based on keyword matching in title + abstract.
No ML models are modified. This is a purely additive module.
"""

import logging
from typing import List, Dict, Tuple
from .score_utils import SCORING_CONFIG

logger = logging.getLogger(__name__)


class PaperClassification:
    SUPPORTING    = "supporting"
    NEUTRAL       = "neutral"
    CONTRADICTING = "contradicting"


def _text_from_paper(paper: Dict) -> str:
    """Extract searchable text from a paper dict (title + abstract)."""
    title    = paper.get("title",    "") or ""
    abstract = paper.get("abstract", "") or ""
    return (title + " " + abstract).lower()


def classify_paper(paper: Dict) -> str:
    """
    Classify a single paper as supporting / neutral / contradicting.

    Returns one of PaperClassification constants.
    """
    text = _text_from_paper(paper)

    supporting_hits    = sum(1 for kw in SCORING_CONFIG["supporting_keywords"]    if kw in text)
    contradicting_hits = sum(1 for kw in SCORING_CONFIG["contradicting_keywords"] if kw in text)

    if contradicting_hits > 0 and contradicting_hits >= supporting_hits:
        return PaperClassification.CONTRADICTING
    elif supporting_hits > 0:
        return PaperClassification.SUPPORTING
    else:
        return PaperClassification.NEUTRAL


def analyze_papers(papers: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Analyze a list of papers and annotate each with its classification.

    Returns:
        annotated_papers: list of paper dicts with added 'classification' key.
        counts: dict with keys 'supporting', 'neutral', 'contradicting'.
    """
    annotated  = []
    counts     = {
        PaperClassification.SUPPORTING:    0,
        PaperClassification.NEUTRAL:       0,
        PaperClassification.CONTRADICTING: 0,
    }

    for paper in papers:
        try:
            label = classify_paper(paper)
            annotated_paper = {**paper, "classification": label}
            annotated.append(annotated_paper)
            counts[label] += 1
        except Exception as e:
            logger.warning(f"Failed to classify paper '{paper.get('title', 'N/A')}': {e}")
            annotated.append({**paper, "classification": PaperClassification.NEUTRAL})
            counts[PaperClassification.NEUTRAL] += 1

    logger.info(
        f"Paper analysis complete — Supporting: {counts['supporting']}, "
        f"Neutral: {counts['neutral']}, Contradicting: {counts['contradicting']}"
    )
    return annotated, counts
