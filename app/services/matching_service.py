import numpy as np

class MatchingService:

    def calculate_similarity(self, embedding1: list, embedding2: list) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)

        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        similarity = dot_product / (norm_a * norm_b)

        return float(similarity)