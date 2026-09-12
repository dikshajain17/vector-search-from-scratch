import numpy as np
from collections import defaultdict

class LSHIndex:
    def __init__(self, dim, num_tables=4, num_bits=10, seed=42):
        rng = np.random.default_rng(seed)
        self.num_tables = num_tables
        self.planes = [rng.normal(size=(num_bits, dim)) for _ in range(num_tables)]
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        self.store = {}
        self.deleted = set()

    def _hash(self, vector, t):
        proj = self.planes[t] @ vector
        return tuple((proj > 0).astype(int))

    def insert(self, vec_id, vector):
        vector = vector / np.linalg.norm(vector)
        self.store[vec_id] = vector
        for t in range(self.num_tables):
            self.tables[t][self._hash(vector, t)].append(vec_id)

    def delete(self, vec_id):
        self.deleted.add(vec_id)

    def search(self, query, k=5):
        query = query / np.linalg.norm(query)
        candidates = set()
        for t in range(self.num_tables):
            candidates.update(self.tables[t][self._hash(query, t)])
        candidates -= self.deleted
        if not candidates:
            return []
        cand_list = list(candidates)
        cand_vecs = np.array([self.store[c] for c in cand_list])
        sims = cand_vecs @ query
        order = np.argsort(-sims)[:k]
        return [(cand_list[i], float(sims[i])) for i in order]