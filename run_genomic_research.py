"""
run_genomic_research.py
========================
Master Entry Point for the Genomic Explainability, Robustness & Scientific Evidence Framework.

Usage:
    python run_genomic_research.py --all
    python run_genomic_research.py --xai-consensus
    python run_genomic_research.py --perturbation
    python run_genomic_research.py --stability
    python run_genomic_research.py --fidelity
    python run_genomic_research.py --evidence
    python run_genomic_research.py --species GALLUS
    python run_genomic_research.py --species ALL

Fixed Random Seed: 42
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GenomicResearch")


def main():
    parser = argparse.ArgumentParser(
        description="Genomic Explainability, Robustness & Scientific Evidence Framework Entry Point"
    )
    parser.add_argument("--all", action="store_true", help="Run all 5 modules + Genomic Trust Engine")
    parser.add_argument("--xai-consensus", action="store_true", help="Run Module 1: XAI Consensus Analysis")
    parser.add_argument("--perturbation", action="store_true", help="Run Module 2: Perturbation Analysis")
    parser.add_argument("--stability", action="store_true", help="Run Module 3: Explanation Stability")
    parser.add_argument("--fidelity", action="store_true", help="Run Module 4: XAI Fidelity Test")
    parser.add_argument("--evidence", action="store_true", help="Run Module 5: Evidence-Backed Explanation")
    parser.add_argument("--species", type=str, default="ALL", help="Species filter (e.g., GALLUS, DUCK, TURKEY, ALL)")

    args = parser.parse_args()

    run_everything = args.all or not any([
        args.xai_consensus, args.perturbation, args.stability, args.fidelity, args.evidence
    ])

    species_filter = args.species.upper().replace("GALLUS_GALLUS", "GALLUS")

    print("\n" + "=" * 70)
    print("  GENOMIC EXPLAINABILITY & SCIENTIFIC EVIDENCE FRAMEWORK")
    print(f"  Random Seed: {RANDOM_SEED} | Species Filter: {species_filter}")
    print("=" * 70 + "\n")

    # Module 1: XAI Consensus
    if run_everything or args.xai_consensus:
        from src.research.xai_consensus import run_xai_consensus
        logger.info("Executing Module 1: XAI Consensus Analysis...")
        run_xai_consensus(species_filter=species_filter)

    # Module 2: Perturbation Analysis
    if run_everything or args.perturbation:
        from src.research.perturbation_analysis import run_perturbation_analysis
        logger.info("Executing Module 2: Perturbation Analysis...")
        run_perturbation_analysis(species_filter=species_filter)

    # Module 3: Explanation Stability
    if run_everything or args.stability:
        from src.research.explanation_stability import run_explanation_stability
        logger.info("Executing Module 3: Explanation Stability Analysis...")
        run_explanation_stability(species_filter=species_filter)

    # Module 4: XAI Fidelity Test
    if run_everything or args.fidelity:
        from src.research.xai_fidelity import run_xai_fidelity
        logger.info("Executing Module 4: XAI Fidelity Test...")
        run_xai_fidelity(species_filter=species_filter)

    # Module 5: Evidence-Backed Explanation
    if run_everything or args.evidence:
        from src.research.evidence_explanation import run_evidence_explanation
        logger.info("Executing Module 5: Evidence-Backed Explanation...")
        run_evidence_explanation(species_filter=species_filter)

    # Unified Genomic Trust Engine
    if run_everything:
        from src.research.genomic_trust_engine import run_genomic_trust_engine
        logger.info("Executing Unified Genomic Trust Engine...")
        run_genomic_trust_engine()

    print("\n" + "=" * 70)
    print("  FRAMEWORK EXECUTION COMPLETE")
    print("  All outputs saved to outputs/research/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
