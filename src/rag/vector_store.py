import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # L2 distance index
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_papers(self, embeddings: np.ndarray, papers: List[Dict]):
        if len(embeddings) != len(papers):
            raise ValueError("Embeddings and papers must have the same length.")
        if embeddings.shape[0] == 0:
            return
        
        # Add to index
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.metadata.extend(papers)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(np.ascontiguousarray(query_embedding, dtype=np.float32), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                paper = self.metadata[idx].copy()
                # Simple conversion from distance to similarity score for representation
                paper["similarity_score"] = float(1.0 / (1.0 + dist))
                results.append(paper)
                
        return results
