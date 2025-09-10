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
from cuvs.neighbors import ivf_flat

# Initialize resources
resources = Resources()

# Parameters
n_samples = 10000
n_dim = 3
n_queries = 10
n_lists = int(math.sqrt(n_samples))
topk = 10
n_probes = n_lists // 5

# Generate synthetic dataset and queries
dataset = cp.random.random_sample((n_samples, n_dim), dtype=cp.float32)
queries = cp.random.random_sample((n_queries, n_dim), dtype=cp.float32)

# Build IVF Flat index
index_params = ivf_flat.IndexParams(
    n_lists=n_lists, metric="sqeuclidean", kmeans_trainset_fraction=0.1
)
index = ivf_flat.build(index_params, dataset, resources=resources)

# Search the index
search_params = ivf_flat.SearchParams(n_probes=n_probes)
distances, neighbors = ivf_flat.search(
    search_params, index, queries, topk, resources=resources
)

# Sync device
resources.sync()

# Convert results to CuPy arrays
neighbors = cp.asarray(neighbors)
distances = cp.asarray(distances)

# Display results
print("\nSimple Search Results:")
print("=" * 100)
print("Nearest neighbors:\n", neighbors)
print("Distances:\n", distances)

n_train = n_samples // 10

# Sample n_train points from dataset without replacement
indices = cp.random.choice(n_samples, size=n_train, replace=False)
trainset = dataset[indices]

# Build IVF Flat index using the sampled training set
index_params = ivf_flat.IndexParams(
    n_lists=n_lists, metric="sqeuclidean", add_data_on_build=False
)
index = ivf_flat.build(index_params, trainset, resources=resources)

# Fill index with dataset vector
ivf_flat.extend(
    index, dataset, cp.arange(n_samples, dtype=cp.int64), resources=resources
)

# Search the index
search_params = ivf_flat.SearchParams(n_probes=n_probes)
distances, neighbors = ivf_flat.search(
    search_params, index, queries, topk, resources=resources
)

# Sync device
resources.sync()

# Convert results to CuPy arrays
neighbors = cp.asarray(neighbors)
distances = cp.asarray(distances)

# Display results
print("\nExtend Search Results:")
print("=" * 100)
print("Nearest neighbors:\n", neighbors)
print("Distances:\n", distances)
