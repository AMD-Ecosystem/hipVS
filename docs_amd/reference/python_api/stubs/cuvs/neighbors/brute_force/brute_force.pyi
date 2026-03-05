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
from cuvs.neighbors.filters.filters import no_filter
import numpy as np
from pylibraft.common.cai_wrapper import cai_wrapper
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['DISTANCE_TYPES', 'Index', 'auto_convert_output', 'auto_sync_resources', 'build', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'load', 'no_filter', 'np', 'save', 'search', 'wrap_array']
class Index:
    """
    
        Brute Force index object. This object stores the trained Brute Force
        which can be used to perform nearest neighbors searches.
        
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
    def __repr__(*args, **kwargs):
        ...
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Index.__setstate_cython__(self, __pyx_state)
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
    build(dataset, metric=u'sqeuclidean', metric_arg=2.0, resources=None)
    
        Build the Brute Force index from the dataset for efficient search.
    
        Parameters
        ----------
        dataset : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, float16]
        metric : Distance metric to use. Default is sqeuclidean
        metric_arg : value of 'p' for Minkowski distances
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index: cuvs.neighbors.brute_force.Index
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> from cuvs.neighbors import brute_force
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> index = brute_force.build(dataset, metric="cosine")
        >>> distances, neighbors = brute_force.search(index, dataset, k)
        >>> distances = cp.asarray(distances)
        >>> neighbors = cp.asarray(neighbors)
        
    """
def load(*args, resources = None, **kwargs):
    """
    load(filename, resources=None)
    
        Loads index from file.
    
        The serialization format can be subject to changes, therefore loading
        an index saved with a previous version of cuvs is not guaranteed
        to work.
    
    
        Parameters
        ----------
        filename : string
            Name of the file.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index : Index
    
        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import brute_force
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = brute_force.build(dataset)
        >>> # Serialize and deserialize the brute_force index built
        >>> brute_force.save("my_index.bin", index)
        >>> index_loaded = brute_force.load("my_index.bin")
        
    """
def save(*args, resources = None, **kwargs):
    """
    save(filename, Index index, bool include_dataset=True, resources=None)
    
        Saves the index to a file.
    
        The serialization format can be subject to changes, therefore loading
        an index saved with a previous version of cuvs is not guaranteed
        to work.
    
        Parameters
        ----------
        filename : string
            Name of the file.
        index : Index
            Trained Brute Force index.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import brute_force
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = brute_force.build(dataset)
        >>> # Serialize and deserialize the brute_force index built
        >>> brute_force.save("my_index.bin", index)
        >>> index_loaded = brute_force.load("my_index.bin")
        
    """
def search(*args, resources = None, **kwargs):
    """
    search(Index index, queries, k, neighbors=None, distances=None, resources=None, prefilter=None)
    
        Find the k nearest neighbors for each query.
    
        Parameters
        ----------
        index : Index
            Trained Brute Force index.
        queries : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, float16]
        k : int
            The number of neighbors.
        neighbors : Optional CUDA array interface compliant matrix shape
                    (n_queries, k), dtype int64_t. If supplied, neighbor
                    indices will be written here in-place. (default None)
        distances : Optional CUDA array interface compliant matrix shape
                    (n_queries, k) If supplied, the distances to the
                    neighbors will be written here in-place. (default None)
        prefilter : Optional, cuvs.neighbors.cuvsFilter
                    An optional filter to exclude certain query-neighbor
                    pairs using a bitmap or bitset. The filter function should
                    have a row-major layout with logical shape
                    `(n_prefilter_rows, n_samples)`, where:
                    - `n_prefilter_rows == n_queries` when using a bitmap filter.
                    - `n_prefilter_rows == 1` when using a bitset prefilter.
                    Each bit in `n_samples` determines whether `queries[i]`
                    should be considered for distance computation with the index.
                    (default None)
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
    
        >>> # Example without pre-filter
        >>> import cupy as cp
        >>> from cuvs.neighbors import brute_force
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = brute_force.build(dataset, metric="sqeuclidean")
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> k = 10
        >>> # Using a pooling allocator reduces overhead of temporary array
        >>> # creation during search. This is useful if multiple searches
        >>> # are performed with same query size.
        >>> distances, neighbors = brute_force.search(index, queries, k)
        >>> neighbors = cp.asarray(neighbors)
        >>> distances = cp.asarray(distances)
    
        >>> # Example with pre-filter
        >>> import numpy as np
        >>> import cupy as cp
        >>> from cuvs.neighbors import brute_force, filters
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = brute_force.build(dataset, metric="sqeuclidean")
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build filters
        >>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)
        >>> # Create your own bitmap as the filter by replacing the random one.
        >>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)
        >>> bitmap_prefilter = filters.from_bitmap(bitmap)
        >>>
        >>> # or Build bitset prefilter:
        >>> # n_bitset = np.ceil(n_samples * 1 / 32).astype(int)
        >>> # # Create your own bitset as the filter by replacing the random one.
        >>> # bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)
        >>> # bitset_prefilter = filters.from_bitset(bitset)
        >>>
        >>> k = 10
        >>> # Using a pooling allocator reduces overhead of temporary array
        >>> # creation during search. This is useful if multiple searches
        >>> # are performed with same query size.
        >>> distances, neighbors = brute_force.search(index, queries, k,
        ...                                           prefilter=bitmap_prefilter)
        >>> neighbors = cp.asarray(neighbors)
        >>> distances = cp.asarray(distances)
    
        
    """
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 78)': '\n    Build the Brute Force index from the dataset for efficient search.\n\n    Parameters\n    ----------\n    dataset : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, float16]\n    metric : Distance metric to use. Default is sqeuclidean\n    metric_arg : value of \'p\' for Minkowski distances\n    {resources_docstring}\n\n    Returns\n    -------\n    index: cuvs.neighbors.brute_force.Index\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import brute_force\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> index = brute_force.build(dataset, metric="cosine")\n    >>> distances, neighbors = brute_force.search(index, dataset, k)\n    >>> distances = cp.asarray(distances)\n    >>> neighbors = cp.asarray(neighbors)\n    ', 'search (line 137)': '\n    Find the k nearest neighbors for each query.\n\n    Parameters\n    ----------\n    index : Index\n        Trained Brute Force index.\n    queries : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, float16]\n    k : int\n        The number of neighbors.\n    neighbors : Optional CUDA array interface compliant matrix shape\n                (n_queries, k), dtype int64_t. If supplied, neighbor\n                indices will be written here in-place. (default None)\n    distances : Optional CUDA array interface compliant matrix shape\n                (n_queries, k) If supplied, the distances to the\n                neighbors will be written here in-place. (default None)\n    prefilter : Optional, cuvs.neighbors.cuvsFilter\n                An optional filter to exclude certain query-neighbor\n                pairs using a bitmap or bitset. The filter function should\n                have a row-major layout with logical shape\n                `(n_prefilter_rows, n_samples)`, where:\n                - `n_prefilter_rows == n_queries` when using a bitmap filter.\n                - `n_prefilter_rows == 1` when using a bitset prefilter.\n                Each bit in `n_samples` determines whether `queries[i]`\n                should be considered for distance computation with the index.\n                (default None)\n    {resources_docstring}\n\n    Examples\n    --------\n\n    >>> # Example without pre-filter\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import brute_force\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = brute_force.build(dataset, metric="sqeuclidean")\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 10\n    >>> # Using a pooling allocator reduces overhead of temporary array\n    >>> # creation during search. This is useful if multiple searches\n    >>> # are performed with same query size.\n    >>> distances, neighbors = brute_force.search(index, queries, k)\n    >>> neighbors = cp.asarray(neighbors)\n    >>> distances = cp.asarray(distances)\n\n    >>> # Example with pre-filter\n    >>> import numpy as np\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import brute_force, filters\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = brute_force.build(dataset, metric="sqeuclidean")\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build filters\n    >>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)\n    >>> # Create your own bitmap as the filter by replacing the random one.\n    >>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)\n    >>> bitmap_prefilter = filters.from_bitmap(bitmap)\n    >>>\n    >>> # or Build bitset prefilter:\n    >>> # n_bitset = np.ceil(n_samples * 1 / 32).astype(int)\n    >>> # # Create your own bitset as the filter by replacing the random one.\n    >>> # bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)\n    >>> # bitset_prefilter = filters.from_bitset(bitset)\n    >>>\n    >>> k = 10\n    >>> # Using a pooling allocator reduces overhead of temporary array\n    >>> # creation during search. This is useful if multiple searches\n    >>> # are performed with same query size.\n    >>> distances, neighbors = brute_force.search(index, queries, k,\n    ...                                           prefilter=bitmap_prefilter)\n    >>> neighbors = cp.asarray(neighbors)\n    >>> distances = cp.asarray(distances)\n\n    ', 'save (line 284)': '\n    Saves the index to a file.\n\n    The serialization format can be subject to changes, therefore loading\n    an index saved with a previous version of cuvs is not guaranteed\n    to work.\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    index : Index\n        Trained Brute Force index.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import brute_force\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = brute_force.build(dataset)\n    >>> # Serialize and deserialize the brute_force index built\n    >>> brute_force.save("my_index.bin", index)\n    >>> index_loaded = brute_force.load("my_index.bin")\n    ', 'load (line 322)': '\n    Loads index from file.\n\n    The serialization format can be subject to changes, therefore loading\n    an index saved with a previous version of cuvs is not guaranteed\n    to work.\n\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    {resources_docstring}\n\n    Returns\n    -------\n    index : Index\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import brute_force\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = brute_force.build(dataset)\n    >>> # Serialize and deserialize the brute_force index built\n    >>> brute_force.save("my_index.bin", index)\n    >>> index_loaded = brute_force.load("my_index.bin")\n    '}
