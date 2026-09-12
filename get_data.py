import numpy as np
from sentence_transformers import SentenceTransformer
topics = ["cricket match score", "python programming tutorial", "delicious pasta recipe",
          "stock market news", "machine learning basics", "travel to japan", "yoga for beginners",
          "climate change effects", "smartphone review", "history of india"]

texts = []
for t in topics:
    for i in range(500):
        texts.append(f"{t} variant {i} discussion point {i % 7}")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
np.save("data/embeddings.npy", embeddings)

with open("data/texts.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")

print("Saved", embeddings.shape[0], "texts with vector size", embeddings.shape[1])