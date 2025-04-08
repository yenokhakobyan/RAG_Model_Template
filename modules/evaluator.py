from sentence_transformers import SentenceTransformer, util

class Evaluator:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def similarity_score(self, generated: str, reference: str) -> float:
        emb1 = self.model.encode(generated, convert_to_tensor=True)
        emb2 = self.model.encode(reference, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2)
        return float(sim[0][0])
