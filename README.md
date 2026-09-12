# LSH Vector Search — Built from Scratch (numpy only)

I built a small vector database from scratch, using only numpy for the math — no Pinecone, FAISS, Chroma, or sklearn.neighbors. The goal was to actually understand what happens inside a vector database instead of just importing one.

## What's in here

- **`exact_index.py`** — a brute-force index. For any query, it checks the similarity against *every* stored vector and returns the true top matches. It's slow, but it's always correct, so I use it as the ground truth to check everything else against.
- **`lsh_index.py`** — an approximate index I built using Locality-Sensitive Hashing (LSH). Instead of checking every vector, it groups similar vectors into buckets ahead of time, so a search only has to look inside the query's own bucket.
- Both support **insert, search, and delete**.

## How LSH actually works

The core trick is surprisingly simple: draw a few random hyperplanes (think of them as random cutting lines) through the vector space. For any vector, you can ask "which side of each line is it on?" — that gives you a string of 0s and 1s, like `1,0,1,1,0`. That bit-string becomes the vector's "bucket address."

Here's why this is useful: two vectors that are genuinely close together in meaning are very likely to land on the *same side* of most random lines, since a line would have to cut precisely between them to separate them. Two unrelated vectors are much more likely to end up on opposite sides of at least a few lines. So vectors with the same (or very similar) bit-string tend to actually be similar in meaning.

At search time, I only compare the query against vectors that hashed into the same bucket not the whole dataset. I also use several independent hash tables at once, so a pair of vectors only needs to match in *one* of them to be considered  this recovers some of the accuracy that any single unlucky hyperplane might cost.

## The knob: `num_bits`

This is the one dial that trades speed for accuracy. More bits means more hyperplanes, which means smaller and more precise buckets — fewer candidates to check per query (faster), but a higher chance the true nearest neighbor got hashed into a different bucket by pure chance (less accurate). Fewer bits does the opposite: bigger, sloppier buckets, slower search, but a much better chance of still finding the real answer.

## Results — 500 queries, recall@10 against the exact ground truth, 50,000 vectors

| num_bits | recall@10 | QPS |
|---|---|---|
| 6  | 0.982 | 64.9 |
| 10 | 0.957 | 165.4 |
| 14 | 0.877 | 288.0 |

Going from 6 to 14 bits, recall drops from 98% to about 88%, while speed goes up roughly 4.4x (65 → 288 queries per second). That's the actual tradeoff, measured  not just claimed.

## Why deletion is tricky (and what I did about it)

I didn't build true deletion, and I want to be upfront about why rather than pretend it's not a gap.

When you insert a vector, its ID gets scattered across multiple hash tables  up to `num_tables` different buckets, one per table. To actually remove it, I'd have to search through every bucket in every table to find and strip that one ID out, which is expensive and honestly works against the whole point of hashing in the first place.

So instead, I use a tombstone: deleting a vector just marks its ID as "deleted" in a set, and every search silently filters those out. The vector's data still physically sits in the buckets, but it never shows up in results again. This is the same tradeoff most real hash-based systems make when faced with this exact problem — a working, honest compromise instead of a broken "real" delete function.

## About the data

50,000 short synthetic texts across 10 topics (things like cricket, cooking, travel, tech), turned into 384-dimensional vectors using a pretrained model (`all-MiniLM-L6-v2` from sentence-transformers). Vectors are normalized, so cosine similarity just becomes a simple dot product — cheaper to compute and simpler to reason about.

## Running it yourself

pip install numpy sentence-transformers matplotlib pandas
python get_data.py              # builds the corpus and embeddings (50,000 vectors)
python eval.py                  # benchmarks recall and speed at 3 settings
python demo.py                  # type a sentence, get the closest matches back
python test_insert_delete.py    # proves insert and delete actually work

## What it looks like in action

I typed `"cricket score today"` — a query that shares essentially no words with anything in the dataset beyond "cricket." It still correctly returned cricket-related entries with high similarity scores. That's the actual proof that this is doing meaning-based search, not keyword matching.
