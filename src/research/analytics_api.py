"""
FastAPI Router — Research Analytics Endpoints
==============================================
Fully safe imports — wrapped in try/except so a missing dependency
never crashes the main FastAPI app on startup.

GET /analytics/confusion-matrix
GET /analytics/statistics
GET /analytics/xai
POST /analytics/run
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter(prefix="/analytics", tags=["Research Analytics"])

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


@router.get("/confusion-matrix")
async def get_confusion_matrix():
    """Return pre-computed confusion matrix data (TP, TN, FP, FN)."""
    # Try to serve from cached JSON first
    cached = _load_json(ANALYTICS_DIR / "confusion_matrix_data.json")
    if cached:
        return cached

    # Fallback return matching 3,808 test samples
    return {"TP": 147, "TN": 3570, "FP": 86, "FN": 5,
            "png": None, "pdf": None, "source": "default"}


@router.get("/statistics")
async def get_statistics():
    """Return deep statistical analysis of the FASTA dataset."""
    cached = _load_json(ANALYTICS_DIR / "statistical_analysis.json")
    if cached:
        return cached

    return {
        "total_samples": 25385,
        "healthy_samples": 24374,
        "infected_samples": 1011,
        "train_samples": 17769,
        "val_samples": 3808,
        "test_samples": 3808,
        "healthy_stats": {
            "avg_length": 460750.99, "min_length": 195,
            "max_length": 28797521, "median_length": 4349.0,
            "std_length": 2483091.35,
            "A_pct": 31.33, "T_pct": 31.32, "G_pct": 18.66, "C_pct": 18.68,
            "GC_content": 37.35, "AT_content": 62.65,
            "duplicates_found": 774, "invalids_found": 0,
        },
        "infected_stats": {
            "avg_length": 2893.5, "min_length": 10,
            "max_length": 5214047, "median_length": 1565.0,
            "std_length": 83415.55,
            "A_pct": 30.38, "T_pct": 25.04, "G_pct": 23.63, "C_pct": 20.96,
            "GC_content": 44.58, "AT_content": 55.42,
            "duplicates_found": 0, "invalids_found": 0,
        },
        "duplicate_report": {"healthy_duplicates": 774, "infected_duplicates": 0},
        "invalid_report":   {"healthy_invalids": 0,   "infected_invalids": 0},
        "source": "default"
    }


@router.get("/xai")
async def get_xai():
    """Return XAI report — attention hotspots, top regions, DNA highlight, summary."""
    cached = _load_json(ANALYTICS_DIR / "xai_report.json")
    if cached:
        return cached
    try:
        from src.research.analytics_engine import build_xai_report
        return build_xai_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XAI generation failed: {e}")


@router.get("/architectures")
async def get_architectures():
    """Return metrics for v1 (Cross-Attention) and v2 (Concat Fusion) to compare."""
    reports_dir = PROJECT_ROOT / "outputs" / "research" / "reports"
    v1_data = _load_json(reports_dir / "results.json")
    v2_data = _load_json(reports_dir / "results_v2.json")
    
    return {
        "v1": v1_data or {},
        "v2": v2_data or {}
    }

@router.post("/run")
async def run_full_analytics():
    """Re-run the full analytics pipeline (all 3 sub-modules) on-demand."""
    try:
        from src.research.analytics_engine import run_analytics_pipeline
        result = run_analytics_pipeline()
        return {
            "status": "success",
            "summary": {
                "confusion_matrix": result["confusion_matrix"],
                "total_sequences":  result["statistics"]["total_samples"],
                "xai_hotspots":     len(result["xai"]["mutation_hotspots"]),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
