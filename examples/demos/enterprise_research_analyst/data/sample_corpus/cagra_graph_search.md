# CAGRA: GPU-Accelerated Graph-Based Search

CAGRA (Cuda Anns GRAph-based) is a state-of-the-art graph-based approximate nearest-neighbor algorithm designed from the ground up for GPU execution. It was introduced by the NVIDIA RAPIDS team and has been ported to AMD GPUs via hipVS.

## How CAGRA Works

CAGRA builds a k-nearest-neighbor graph over the dataset, then prunes and optimises the graph for efficient traversal. At search time, the algorithm performs a greedy walk through the graph, visiting neighbours of neighbours until convergence.

## GPU-Friendly Design

Unlike CPU-oriented graph algorithms (e.g., HNSW), CAGRA's traversal is designed for GPU parallelism. Multiple search queries are batched and executed simultaneously, saturating GPU compute and memory bandwidth. This yields orders-of-magnitude speedup over CPU baselines.

## Build and Search Parameters

Key build parameters include graph_degree (edges per node) and intermediate_graph_degree (used during construction for higher quality). Search parameters like itopk_size control the trade-off between recall and latency.

## CAGRA + HNSW Hybrid

CAGRA indexes can be exported to HNSW format for CPU-based serving, enabling a workflow where indexes are built on GPU (fast) and served on CPU (cost-effective, no GPU needed at query time).
