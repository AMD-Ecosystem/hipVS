# Vector Search and Similarity

Vector search (also called similarity search or nearest-neighbor search) is the process of finding data points in a high-dimensional space that are closest to a given query vector. It is the backbone of modern information-retrieval systems, recommendation engines, and AI applications.

## Distance Metrics

The choice of distance metric determines how closeness is measured. Common metrics include L2 (Euclidean) distance, cosine similarity, and inner product. For normalised embeddings, cosine similarity and inner product are equivalent. Jaccard and Hamming distances are used for binary or set-valued features.

## Exact vs. Approximate Search

Brute-force search computes the distance between the query and every vector in the database, guaranteeing perfect recall. However, it scales linearly with dataset size. Approximate nearest-neighbor (ANN) algorithms trade a small amount of recall for dramatically lower latency, making billion-scale search practical. Prominent ANN strategies include graph-based methods (HNSW, CAGRA), inverted-file indexes (IVF-Flat, IVF-PQ), and tree-based partitioning (Annoy).

## GPU Acceleration

GPUs excel at vector search because distance computation is inherently parallel: each distance can be computed independently. Libraries like hipVS exploit thousands of GPU cores to achieve sub-millisecond search latency on datasets with millions of vectors.
