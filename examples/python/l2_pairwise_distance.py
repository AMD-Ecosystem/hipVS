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
from cuvs.distance import pairwise_distance

# Initialize resources
resources = Resources()

n_samples = 10
n_features = 5

vecs1 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
vecs2 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)

distances = pairwise_distance(vecs1, vecs2, metric="l2", resources=resources)
resources.sync()

print("Vectors group 1:\n", vecs1)
print("Vectors group 2:\n", vecs2)
print("Distance:\n", cp.asarray(distances))
