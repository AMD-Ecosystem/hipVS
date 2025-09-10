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

import cupy as cp
from cuvs.common import Resources
from cuvs.neighbors import cagra

# Initialize resources
resources = Resources()

# Parameters
n_samples = 10000
n_dim = 90
n_queries = 10
topk = 12

# Generate synthetic dataset and queries
dataset = cp.random.random_sample((n_samples, n_dim), dtype=cp.float32)
queries = cp.random.random_sample((n_queries, n_dim), dtype=cp.float32)

# Build CAGRA index
index_params = cagra.IndexParams()
index = cagra.build(index_params, dataset, resources=resources)

# Search the index
search_params = cagra.SearchParams()
distances, neighbors = cagra.search(
    search_params, index, queries, topk, resources=resources
)

# Sync device to ensure results are ready
resources.sync()

neighbors = cp.asarray(neighbors)
distances = cp.asarray(distances)

# Display results
print("Nearest neighbors:\n", neighbors)
print("Distances:\n", distances)
