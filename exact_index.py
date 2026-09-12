import numpy as np

class ExactIndex:
    def __init__(self, dim):
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids = []
        self.deleted = set()

    def insert(self, vec_id, vector):
        vector = vector / np.linalg.norm(vector)
        self.vectors = np.vstack([self.vectors, vector])
        self.ids.append(vec_id)

    def delete(self, vec_id):
        self.deleted.add(vec_id)

    def search(self, query, k=5):
        query = query / np.linalg.norm(query)
        sims = self.vectors @ query
        order = np.argsort(-sims)
        results = []
        for idx in order:
            vid = self.ids[idx]
            if vid in self.deleted:
                continue
            results.append((vid, float(sims[idx])))
            if len(results) == k:
                break
        return results