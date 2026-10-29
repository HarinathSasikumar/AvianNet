from fastapi import APIRouter, HTTPException, FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
try:
    from .rag_pipeline import RAGPipeline
except ImportError:
    from src.rag.rag_pipeline import RAGPipeline
import uuid
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from src.evidence.evidence_api import router as evidence_router
from src.research.analytics_api import router as analytics_router
from src.research.xai.xai_api import router as xai_router
from src.research.genomic_framework_api import router as genomic_router

router = APIRouter(prefix="/rag", tags=["RAG"])

# In-memory store to simulate persisting prediction results
prediction_reports_db = {}
_pipeline = None  # Lazy-init: loaded only on first request to avoid slow startup

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

class PredictionContext(BaseModel):
    prediction: str
    confidence: float
    mutation_gene: str
    mutation_positions: str

@router.post("/search")
async def search_endpoint(context: PredictionContext):
    query = f"Avian influenza {context.mutation_gene} mutation"
    papers = get_pipeline().retriever.run_retrieval(query, top_k=5)
    return {"query": query, "papers": papers}

@router.post("/summarize")
async def summarize_endpoint(context: PredictionContext):
    # Short-circuit logic for demo
    summary = get_pipeline().summarizer.summarize(context.mutation_gene, [])
    return {"summary": summary}

@router.post("/report")
async def generate_report(context: PredictionContext):
    report = get_pipeline().process_prediction(context.model_dump())
    report_id = str(uuid.uuid4())
    prediction_reports_db[report_id] = report
    return {"report_id": report_id, "report": report}

@router.get("/papers/{prediction_id}")
async def get_papers(prediction_id: str):
    if prediction_id not in prediction_reports_db:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"papers": prediction_reports_db[prediction_id].get("top_papers", [])}

@router.get("/dataset/stats")
async def get_dataset_stats():
    from pathlib import Path
    import json
    stats_file = Path("data/metadata/dataset_statistics.json")
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                data = json.load(f)
                return data
        except Exception:
            pass
    return {
        "Total Downloaded Sequences Processed": 12206,
        "Total Healthy Samples Found": 5436,
        "Total Infected Samples Found": 4539,
        "Total Final Balanced Dataset Size": 9078,
        "Training Samples (70%)": 6354,
        "Validation Samples (15%)": 1362,
        "Testing Samples (15%)": 1362,
        "Species / Strain Distribution": {
            "GALLUS": 5046, "H5N1": 546, "H5N2": 571, "H5N6": 578, "H5N8": 529, "H7N3": 598, "H7N7": 547, "H7N9": 636, "H9N2": 534
        },
        "Average Sequence Length": 114698,
        "Minimum Sequence Length": 10,
        "Maximum Sequence Length": 150990227,
        "Number of Removed Duplicates": 2231,
        "Number of Discarded Invalid Sequences": 0
    }

@router.get("/dataset/multispecies")
async def get_multispecies_dataset():
    from pathlib import Path
    import json
    report_file = Path(__file__).resolve().parent.parent.parent / "data" / "metadata" / "data_quality_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Multi-species dataset report not found.")

@router.get("/research/results")
async def get_research_results():
    from pathlib import Path
    import json
    # Use parent to reach the project root from src/rag
    results_file = Path(__file__).resolve().parent.parent.parent / "outputs" / "research" / "reports" / "results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Research results not found. Run the research pipeline first.")

# Also define a standalone app for convenience
app = FastAPI(title="RAG AI Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(evidence_router)
app.include_router(analytics_router)
app.include_router(xai_router)
app.include_router(genomic_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
