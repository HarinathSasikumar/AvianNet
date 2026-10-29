from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        # Returns embeddings as numpy array of shape (N, D)
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
