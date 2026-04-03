from sentence_transformers import SentenceTransformer

import numpy as np

class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def generate_embedding(self, text: str):
        embedding = self.model.encode(text)
        return embedding.tolist()