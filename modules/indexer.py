import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def build(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_vector: np.ndarray, top_k=3):
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0]
