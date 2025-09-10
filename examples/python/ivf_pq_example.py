# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import cupy as cp
from cuvs.common import Resources
from cuvs.neighbors import ivf_pq, refine

# Initialize resources
resources = Resources()

# Parameters
n_samples = 10000
n_dim = 3
n_queries = 10
topk = 10
topk_refined = 7
n_lists = int(math.sqrt(n_samples))
n_probes = n_lists // 5
pq_bits = 8
pq_dim = 2

# Generate synthetic dataset and queries
dataset = cp.random.random_sample((n_samples, n_dim), dtype=cp.float32)
queries = cp.random.random_sample((n_queries, n_dim), dtype=cp.float32)

# Build IVF-PQ index
index_params = ivf_pq.IndexParams(
    n_lists=n_lists,
    kmeans_trainset_fraction=0.1,
    metric="sqeuclidean",
    pq_bits=pq_bits,
    pq_dim=pq_dim,
)
index = ivf_pq.build(index_params, dataset, resources=resources)

# Search the index
search_params = ivf_pq.SearchParams(
    n_probes=n_probes, internal_distance_dtype=cp.float16, lut_dtype=cp.float16
)
distances, neighbors = ivf_pq.search(
    search_params, index, queries, topk, resources=resources
)

# Refine the results using exact distances
refined_distances, refined_neighbors = refine(
    dataset=dataset,
    queries=queries,
    candidates=neighbors,
    k=topk_refined,
    metric=index_params.metric,
    resources=resources,
)

# Sync device
resources.sync()

# Convert results to CuPy arrays
neighbors = cp.asarray(neighbors)
distances = cp.asarray(distances)
refined_neighbors = cp.asarray(refined_neighbors)
refined_distances = cp.asarray(refined_distances)

# Display results
print("Original IVF-PQ Search Results:")
print("=" * 100)
print("Nearest neighbors:\n", neighbors)
print("Distances:\n", distances)

print("\nRefined Results (Exact Distance Re-ranking):")
print("=" * 100)
print("Nearest neighbors:\n", refined_neighbors)
print("Distances:\n", refined_distances)
