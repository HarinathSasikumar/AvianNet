from .retriever import Retriever
from .llm_summarizer import LLMSummarizer
from .recommendation_engine import RecommendationEngine
from typing import Dict, Any

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.summarizer = LLMSummarizer()
        self.recommender = RecommendationEngine()

    def process_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline given a prediction context.
        prediction_data expects:
        {
            "prediction": "Infected",
            "confidence": 0.986,
            "mutation_gene": "HA Gene",
            "mutation_positions": "125-138"
        }
        """
        gene = prediction_data.get("mutation_gene", "Avian Influenza")
        
        # Generate search query
        query = f"Avian influenza {gene} mutation"
        
        # 1. Retrieve Papers
        top_papers = self.retriever.run_retrieval(query, top_k=5)
        
        # 2. Generate Summary
        mutation_context = f"Gene: {gene}, Positions: {prediction_data.get('mutation_positions', 'Unknown')}"
        summary = self.summarizer.summarize(mutation_context, top_papers)
        
        # 3. Generate Recommendations
        recommendations = self.recommender.generate_recommendations(gene)
        
        return {
            "prediction": prediction_data.get("prediction"),
            "confidence": prediction_data.get("confidence"),
            "mutation": gene,
            "position": prediction_data.get("mutation_positions"),
            "top_papers": top_papers,
            "scientific_evidence": summary,
            "recommendations": recommendations,
            "disclaimer": "AI recommendations are for research support only and do not constitute a medical diagnosis."
        }
