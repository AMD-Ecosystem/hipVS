# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
import builtins as __builtins__
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['DISTANCE_TYPES', 'auto_convert_output', 'auto_sync_resources', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'np', 'refine', 'wrap_array']
def refine(*args, resources = None, **kwargs):
    """
    refine(dataset, queries, candidates, k=None, metric='sqeuclidean', indices=None, distances=None, resources=None)
    
    Refine nearest neighbor search.
    
    Refinement is an operation that follows an approximate NN search. The
    approximate search has already selected n_candidates neighbor candidates
    for each query. We narrow it down to k neighbors. For each query, we
    calculate the exact distance between the query and its n_candidates
    neighbor candidate, and select the k nearest ones.
    
    Input arrays can be either CUDA array interface compliant matrices or
    array interface compliant matrices in host memory. All array must be in
    the same memory space.
    
    Parameters
    ----------
    dataset : array interface compliant matrix, shape (n_samples, dim)
        Supported dtype [float32, int8, uint8, float16]
    queries : array interface compliant matrix, shape (n_queries, dim)
        Supported dtype [float32, int8, uint8, float16]
    candidates : array interface compliant matrix, shape (n_queries, k0)
        Supported dtype int64
    k : int
        Number of neighbors to search (k <= k0). Optional if indices or
        distances arrays are given (in which case their second dimension
        is k).
    metric : str
        Name of distance metric to use, default ="sqeuclidean"
    indices :  Optional array interface compliant matrix shape                (n_queries, k).
        If supplied, neighbor indices will be written here in-place.
        (default None). Supported dtype int64.
    distances :  Optional array interface compliant matrix shape                  (n_queries, k).
        If supplied, neighbor indices will be written here in-place.
        (default None) Supported dtype float.
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.common import Resources
    >>> from cuvs.neighbors import ivf_pq, refine
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> resources = Resources()
    >>> index_params = ivf_pq.IndexParams(n_lists=1024,
    ...                                   metric="sqeuclidean",
    ...                                   pq_dim=10)
    >>> index = ivf_pq.build(index_params, dataset, resources=resources)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 40
    >>> _, candidates = ivf_pq.search(ivf_pq.SearchParams(), index,
    ...                               queries, k, resources=resources)
    >>> k = 10
    >>> distances, neighbors = refine(dataset, queries, candidates, k)
    """
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'refine (line 32)': '\n    Refine nearest neighbor search.\n\n    Refinement is an operation that follows an approximate NN search. The\n    approximate search has already selected n_candidates neighbor candidates\n    for each query. We narrow it down to k neighbors. For each query, we\n    calculate the exact distance between the query and its n_candidates\n    neighbor candidate, and select the k nearest ones.\n\n    Input arrays can be either CUDA array interface compliant matrices or\n    array interface compliant matrices in host memory. All array must be in\n    the same memory space.\n\n    Parameters\n    ----------\n    dataset : array interface compliant matrix, shape (n_samples, dim)\n        Supported dtype [float32, int8, uint8, float16]\n    queries : array interface compliant matrix, shape (n_queries, dim)\n        Supported dtype [float32, int8, uint8, float16]\n    candidates : array interface compliant matrix, shape (n_queries, k0)\n        Supported dtype int64\n    k : int\n        Number of neighbors to search (k <= k0). Optional if indices or\n        distances arrays are given (in which case their second dimension\n        is k).\n    metric : str\n        Name of distance metric to use, default ="sqeuclidean"\n    indices :  Optional array interface compliant matrix shape                (n_queries, k).\n        If supplied, neighbor indices will be written here in-place.\n        (default None). Supported dtype int64.\n    distances :  Optional array interface compliant matrix shape                  (n_queries, k).\n        If supplied, neighbor indices will be written here in-place.\n        (default None) Supported dtype float.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.common import Resources\n    >>> from cuvs.neighbors import ivf_pq, refine\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> resources = Resources()\n    >>> index_params = ivf_pq.IndexParams(n_lists=1024,\n    ...                                   metric="sqeuclidean",\n    ...                                   pq_dim=10)\n    >>> index = ivf_pq.build(index_params, dataset, resources=resources)\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 40\n    >>> _, candidates = ivf_pq.search(ivf_pq.SearchParams(), index,\n    ...                               queries, k, resources=resources)\n    >>> k = 10\n    >>> distances, neighbors = refine(dataset, queries, candidates, k)\n    '}
