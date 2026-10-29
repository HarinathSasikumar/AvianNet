"""Evidence module init"""
from .trust_score import generate_full_evidence_report
from .evidence_score import calculate_evidence_score, generate_evidence_explanation
from .paper_analyzer import analyze_papers
from .score_utils import SCORING_CONFIG
