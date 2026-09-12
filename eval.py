import numpy as np
import time
from exact_index import ExactIndex
from lsh_index import LSHIndex

embeddings = np.load("data/embeddings.npy")
dim = embeddings.shape[1]

exact = ExactIndex(dim)
for i, v in enumerate(embeddings):
    exact.insert(i, v)

rng = np.random.default_rng(0)
query_idxs = rng.choice(len(embeddings), size=500, replace=False)

ground_truth = {}
for qi in query_idxs:
    results = exact.search(embeddings[qi], k=10)
    ground_truth[int(qi)] = [r[0] for r in results]

def recall_at_k(retrieved, truth, k=10):
    return len(set(retrieved[:k]) & set(truth[:k])) / k

def run_setting(num_bits):
    lsh = LSHIndex(dim, num_tables=4, num_bits=num_bits)
    for i, v in enumerate(embeddings):
        lsh.insert(i, v)

    recalls = []
    start = time.time()
    for qi in query_idxs:
        retrieved = [r[0] for r in lsh.search(embeddings[qi], k=10)]
        recalls.append(recall_at_k(retrieved, ground_truth[int(qi)]))
    elapsed = time.time() - start

    avg_recall = float(np.mean(recalls))
    qps = len(query_idxs) / elapsed
    return avg_recall, qps

print(f"{'num_bits':>10} {'recall@10':>12} {'QPS':>10}")
for bits in [6, 10, 14]:
    recall, qps = run_setting(bits)
    print(f"{bits:>10} {recall:>12.3f} {qps:>10.1f}")