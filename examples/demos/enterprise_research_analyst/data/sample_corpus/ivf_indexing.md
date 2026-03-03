# Inverted File Indexes (IVF)

Inverted File (IVF) indexes partition the vector space into clusters and restrict search to only the most relevant clusters. This dramatically reduces the number of distance computations needed.

## IVF-Flat

IVF-Flat stores full (uncompressed) vectors in each cluster. At search time, the query is compared to cluster centroids, and the n_probes closest clusters are exhaustively searched. IVF-Flat offers exact distance computation within probed clusters, giving high recall at the cost of higher memory usage.

## IVF-PQ (Product Quantization)

IVF-PQ compresses vectors within each cluster using product quantization, which splits each vector into sub-vectors and encodes them with a codebook. This reduces memory by 10-50x at the cost of some recall. It is ideal for very large datasets that must fit in GPU memory.

## Tuning IVF Parameters

The number of IVF lists (n_lists) controls partition granularity. More lists mean smaller clusters and faster search but potentially lower recall. n_probes at search time determines how many clusters are visited -- higher probes improve recall but increase latency.

## When to Choose IVF

IVF-Flat is a good default when memory is not constrained and you need high recall. IVF-PQ is preferred when the dataset is too large for full-precision storage, or when memory-bandwidth is the bottleneck.
