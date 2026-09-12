from sentence_transformers import SentenceTransformer
import numpy as np
from lsh_index import LSHIndex

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("data/embeddings.npy")
texts = open("data/texts.txt", encoding="utf-8").read().splitlines()

lsh = LSHIndex(embeddings.shape[1], num_tables=4, num_bits=10)
for i, v in enumerate(embeddings):
    lsh.insert(i, v)

print("Ready. Type a query (or 'quit'):")
while True:
    q = input("\n> ")
    if q.strip().lower() == "quit":
        break
    qvec = model.encode([q], normalize_embeddings=True)[0]
    for rid, score in lsh.search(qvec, k=5):
        print(f"  [{score:.3f}] {texts[rid]}")