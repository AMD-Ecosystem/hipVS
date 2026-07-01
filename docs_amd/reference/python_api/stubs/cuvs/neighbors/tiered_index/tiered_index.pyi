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
from cuvs.neighbors import cagra
from cuvs.neighbors.common import _check_input_array
from cuvs.neighbors.filters.filters import no_filter
from cuvs.neighbors import ivf_flat
from cuvs.neighbors import ivf_pq
import numpy as np
from pylibraft.common.cai_wrapper import cai_wrapper
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['ALGO_NAMES', 'ALGO_TYPES', 'DISTANCE_NAMES', 'DISTANCE_TYPES', 'Index', 'IndexParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'cagra', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'extend', 'ivf_flat', 'ivf_pq', 'no_filter', 'np', 'search', 'wrap_array']
class Index:
    """
    
        Tiered Index object.
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Index.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Index.__setstate_cython__(self, __pyx_state)
        """
class IndexParams:
    """
    IndexParams(metric='sqeuclidean', *, algo='cagra', upstream_params=None, min_ann_rows=None, create_ann_index_on_extend=None)
    
    Parameters to build index for Tiered Index nearest neighbor search
    
    Parameters
    ----------
    metric : str, default = "sqeuclidean"
        String denoting the metric type.
        Valid values for metric: ["sqeuclidean", "inner_product",
        "euclidean", "cosine"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - euclidean is the euclidean distance
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
            - cosine distance is defined as
              distance(a, b) = 1 - \\sum_i a_i * b_i / ( ||a||_2 * ||b||_2).
    algo : str, default = "cagra"
        The algorithm to use for the ANN portion of the tiered index
    upstream_params : object, optional
        The IndexParams for the upstream ANN object to use (ie the Cagra
        IndexParams for cagra etc)
    min_ann_rows : int
        The minimum number of rows necessary to create an ann index
    create_ann_index_on_extend : bool
        Whether or not to create a new ann index on extend, if the number
        of rows in the incremental (bfknn) portion is above min_ann_rows
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        IndexParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        IndexParams.__setstate_cython__(self, __pyx_state)
        """
def __reduce_cython__(self):
    """
    Index.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    Index.__setstate_cython__(self, __pyx_state)
    """
def build(*args, resources = None, **kwargs):
    """
    build(IndexParams index_params, dataset, resources=None)
    
    Build the Tiered index from the dataset for efficient search.
    
    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.tiered_index.IndexParams`
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32]
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    index: py:class:`cuvs.neighbors.tiered_index.Index`
    
    Examples
    --------
    
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra, tiered_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = tiered_index.IndexParams(metric="sqeuclidean",
    ...                                         algo="cagra")
    >>> index = tiered_index.build(build_params, dataset)
    >>> distances, neighbors = tiered_index.search(cagra.SearchParams(),
    ...                                            index, dataset, k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """
def extend(*args, resources = None, **kwargs):
    """
    extend(Index index, new_vectors, resources=None)
    
    Extend an existing index with new vectors.
    
    The input array can be either CUDA array interface compliant matrix or
    array interface compliant matrix in host memory.
    
    
    Parameters
    ----------
    index : tiered_index.Index
        Trained tiered_index object.
    new_vectors : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32]
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    index: py:class:`cuvs.neighbors.tiered_index.Index`
    
    Examples
    --------
    
    >>> import cupy as cp
    >>> from cuvs.neighbors import tiered_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> index = tiered_index.build(tiered_index.IndexParams(), dataset)
    >>> n_rows = 100
    >>> more_data = cp.random.random_sample((n_rows, n_features),
    ...                                     dtype=cp.float32)
    >>> index = tiered_index.extend(index, more_data)
    """
def search(*args, resources = None, **kwargs):
    """
    search(search_params, Index index, queries, k, neighbors=None, distances=None, resources=None, filter=None)
    
    Find the k nearest neighbors for each query.
    
    Parameters
    ----------
    search_params : SearchParams for the upstream ANN index
    index : py:class:`cuvs.neighbors.tiered_index.Index`
        Trained Tiered index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter
                neighbors based on a given bitset. (default None)
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra, tiered_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build the index
    >>> index = tiered_index.build(tiered_index.IndexParams(algo="cagra"),
    ...                            dataset)
    >>>
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = cagra.SearchParams()
    >>>
    >>> distances, neighbors = tiered_index.search(search_params, index,
    ...                                            queries, k)
    """
ALGO_NAMES: dict = {0: 'cagra', 1: 'ivf_flat', 2: 'ivf_pq'}
ALGO_TYPES: dict = {'cagra': 0, 'ivf_flat': 1, 'ivf_pq': 2}
DISTANCE_NAMES: dict = {1: 'euclidean', 0: 'sqeuclidean', 3: 'cityblock', 6: 'inner_product', 7: 'chebyshev', 8: 'canberra', 2: 'cosine', 9: 'minkowski', 10: 'correlation', 11: 'jaccard', 12: 'hellinger', 14: 'braycurtis', 15: 'jensenshannon', 16: 'hamming', 17: 'kl_divergence', 18: 'russellrao', 19: 'dice', 20: 'bitwise_hamming'}
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 163)': '\n    Build the Tiered index from the dataset for efficient search.\n\n    Parameters\n    ----------\n    index_params : :py:class:`cuvs.neighbors.tiered_index.IndexParams`\n    dataset : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.tiered_index.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra, tiered_index\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = tiered_index.IndexParams(metric="sqeuclidean",\n    ...                                         algo="cagra")\n    >>> index = tiered_index.build(build_params, dataset)\n    >>> distances, neighbors = tiered_index.search(cagra.SearchParams(),\n    ...                                            index, dataset, k)\n    >>> distances = cp.asarray(distances)\n    >>> neighbors = cp.asarray(neighbors)\n    ', 'search (line 221)': '\n    Find the k nearest neighbors for each query.\n\n    Parameters\n    ----------\n    search_params : SearchParams for the upstream ANN index\n    index : py:class:`cuvs.neighbors.tiered_index.Index`\n        Trained Tiered index.\n    queries : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32]\n    k : int\n        The number of neighbors.\n    neighbors : Optional CUDA array interface compliant matrix shape\n                (n_queries, k), dtype int64_t. If supplied, neighbor\n                indices will be written here in-place. (default None)\n    distances : Optional CUDA array interface compliant matrix shape\n                (n_queries, k) If supplied, the distances to the\n                neighbors will be written here in-place. (default None)\n    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter\n                neighbors based on a given bitset. (default None)\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra, tiered_index\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build the index\n    >>> index = tiered_index.build(tiered_index.IndexParams(algo="cagra"),\n    ...                            dataset)\n    >>>\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 10\n    >>> search_params = cagra.SearchParams()\n    >>>\n    >>> distances, neighbors = tiered_index.search(search_params, index,\n    ...                                            queries, k)\n    ', 'extend (line 325)': '\n    Extend an existing index with new vectors.\n\n    The input array can be either CUDA array interface compliant matrix or\n    array interface compliant matrix in host memory.\n\n\n    Parameters\n    ----------\n    index : tiered_index.Index\n        Trained tiered_index object.\n    new_vectors : array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.tiered_index.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import tiered_index\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> index = tiered_index.build(tiered_index.IndexParams(), dataset)\n    >>> n_rows = 100\n    >>> more_data = cp.random.random_sample((n_rows, n_features),\n    ...                                     dtype=cp.float32)\n    >>> index = tiered_index.extend(index, more_data)\n    '}
