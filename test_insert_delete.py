import numpy as np
from lsh_index import LSHIndex

embeddings = np.load("data/embeddings.npy")
dim = embeddings.shape[1]

lsh = LSHIndex(dim, num_tables=4, num_bits=10)
for i, v in enumerate(embeddings):
    lsh.insert(i, v)

# INSERT test: add a brand new vector
new_vec = embeddings[0] + np.random.normal(scale=0.01, size=dim)  # slightly modified copy
new_vec = new_vec / np.linalg.norm(new_vec)
lsh.insert(9999, new_vec)
print("After insert, searching new vector's neighborhood:")
print(lsh.search(new_vec, k=3))

# DELETE test: delete id 0, then search near it, confirm it's gone
print("\nBefore delete, searching near id 0:")
print(lsh.search(embeddings[0], k=3))

lsh.delete(0)
print("\nAfter deleting id 0, searching same query again:")
print(lsh.search(embeddings[0], k=3))