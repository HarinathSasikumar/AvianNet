"""
FastAPI Router — Advanced XAI Endpoints
=========================================
Serves all SHAP / LIME outputs to the dashboard.
Mounted on the existing app with two additive lines — no existing route changes.

GET  /xai/status                → which outputs exist
GET  /xai/shap/results          → SHAP JSON (paths + global importance)
GET  /xai/lime/results          → LIME JSON (explanations + paths)
GET  /xai/image/{filename}      → serve any PNG from outputs/research/xai/
POST /xai/run/shap              → (re-)run SHAP pipeline
POST /xai/run/lime              → (re-)run LIME pipeline
POST /xai/run/all               → run both pipelines
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json

router = APIRouter(prefix="/xai", tags=["Advanced XAI"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
XAI_DIR      = PROJECT_ROOT / "outputs" / "research" / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

# Also expose the existing attention heatmap from research figures
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "research" / "figures"


def _load_json(path: Path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ── Status ─────────────────────────────────────────────────────────

@router.get("/status")
async def xai_status():
    """Returns which XAI output files have been generated."""
    files = {
        "shap_results":    (XAI_DIR / "shap_results.json").exists(),
        "shap_summary":    (XAI_DIR / "shap_summary.png").exists(),
        "shap_importance": (XAI_DIR / "shap_importance.png").exists(),
        "shap_waterfall":  (XAI_DIR / "shap_waterfall.png").exists(),
        "lime_results":    (XAI_DIR / "lime_results.json").exists(),
        "lime_chart":      (XAI_DIR / "lime_explanation.png").exists(),
        "attention_heatmap": (FIGURES_DIR / "attention_heatmap.png").exists(),
    }
    return {"ready": all(files.values()), "files": files}


# ── SHAP ───────────────────────────────────────────────────────────

@router.get("/shap/results")
async def get_shap_results():
    """Return cached SHAP results JSON."""
    data = _load_json(XAI_DIR / "shap_results.json")
    if data:
        # Convert file paths to API image URLs
        for key in ("shap_summary_png", "shap_importance_png", "shap_waterfall_png"):
            if key in data and data[key]:
                data[key.replace("_png", "_url")] = f"/xai/image/{Path(data[key]).name}"
        return data
    raise HTTPException(status_code=404, detail="SHAP results not found. POST to /xai/run/shap first.")


# ── LIME ───────────────────────────────────────────────────────────

@router.get("/lime/results")
async def get_lime_results():
    """Return cached LIME results JSON."""
    data = _load_json(XAI_DIR / "lime_results.json")
    if data:
        if data.get("lime_chart_png"):
            data["lime_chart_url"] = f"/xai/image/{Path(data['lime_chart_png']).name}"
        return data
    raise HTTPException(status_code=404, detail="LIME results not found. POST to /xai/run/lime first.")


# ── Image serving ──────────────────────────────────────────────────

@router.get("/image/{filename}")
async def serve_xai_image(filename: str):
    """Serve a generated XAI PNG from outputs/research/xai/."""
    # Also check the research figures directory for attention heatmap
    for search_dir in (XAI_DIR, FIGURES_DIR, XAI_DIR / "fidelity_figures", XAI_DIR / "consensus_figures"):
        path = search_dir / filename
        if path.exists() and path.suffix in (".png", ".pdf", ".jpg"):
            return FileResponse(str(path), media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Image {filename} not found.")


# ── Run endpoints ──────────────────────────────────────────────────

@router.post("/run/shap")
def run_shap():
    """Trigger SHAP analysis pipeline (may take 30–60 s)."""
    try:
        from src.research.xai.shap_analyzer import run_shap_analysis
        result = run_shap_analysis()
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"status": "success", "files_generated": list(result.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/lime")
def run_lime():
    """Trigger LIME analysis pipeline (may take 60–120 s)."""
    try:
        from src.research.xai.lime_analyzer import run_lime_analysis
        result = run_lime_analysis()
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"status": "success", "explanations": len(result.get("explanations", []))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/all")
def run_all_xai():
    """Run both SHAP and LIME pipelines sequentially."""
    results = {}
    try:
        from src.research.xai.shap_analyzer import run_shap_analysis
        results["shap"] = run_shap_analysis()
    except Exception as e:
        results["shap"] = {"error": str(e)}
    try:
        from src.research.xai.lime_analyzer import run_lime_analysis
        results["lime"] = run_lime_analysis()
    except Exception as e:
        results["lime"] = {"error": str(e)}
    return {"status": "complete", "results": results}

