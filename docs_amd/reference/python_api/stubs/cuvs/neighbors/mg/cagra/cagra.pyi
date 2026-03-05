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
from cuvs.common.mg_resources import auto_sync_multi_gpu_resources
import cuvs.neighbors.cagra.cagra
from cuvs.neighbors.common import _check_input_array
from cuvs.neighbors.common import _check_memory_location
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['Index', 'IndexParams', 'SearchParams', 'auto_convert_output', 'auto_sync_multi_gpu_resources', 'build', 'check_cuvs', 'cuda_interruptible', 'distribute', 'extend', 'load', 'np', 'save', 'search', 'wrap_array']
class Index:
    """
    
        Multi-GPU CAGRA index object. Stores the trained multi-GPU CAGRA index
        state which can be used to perform nearest neighbors searches across
        multiple GPUs.
        
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
class IndexParams(cuvs.neighbors.cagra.cagra.IndexParams):
    """
    IndexParams(distribution_mode=u'sharded', *, **kwargs)
    
        Parameters to build multi-GPU CAGRA index for efficient search.
        Extends single-GPU IndexParams with multi-GPU specific parameters.
    
        Parameters
        ----------
        distribution_mode : str, default = "sharded"
            Distribution mode for multi-GPU setup.
            Valid values: ["replicated", "sharded"]
        **kwargs : Additional parameters passed to single-GPU IndexParams
    
        Note
        ----
        CAGRA currently only supports "sqeuclidean" and "inner_product" metrics.
        
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
    def get_handle(self):
        """
        IndexParams.get_handle(self)
        """
class SearchParams(cuvs.neighbors.cagra.cagra.SearchParams):
    """
    SearchParams(search_mode=u'load_balancer', *, merge_mode=u'merge_on_root_rank', n_rows_per_batch=1000, **kwargs)
    
        Parameters to search multi-GPU CAGRA index.
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        SearchParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        SearchParams.__setstate_cython__(self, __pyx_state)
        """
    def get_handle(self):
        """
        SearchParams.get_handle(self)
        """
def __reduce_cython__(self):
    """
    SearchParams.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    SearchParams.__setstate_cython__(self, __pyx_state)
    """
def build(*args, resources = None, **kwargs):
    """
    build(IndexParams index_params, dataset, resources=None)
    
        Build the multi-GPU CAGRA index from the dataset for efficient search.
    
        Parameters
        ----------
        index_params : :py:class:`cuvs.neighbors.cagra.IndexParams`
        dataset : Array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, float16, int8, uint8]
            **IMPORTANT**: For multi-GPU CAGRA, the dataset MUST be in host
            memory (CPU). If using CuPy/device arrays, transfer to host with
            array.get() or cp.asnumpy(array).
        resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.
            If Multi-GPU Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index: py:class:`cuvs.neighbors.cagra.Index`
    
        Examples
        --------
    
        >>> import numpy as np
        >>> from cuvs.neighbors.mg import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> # For multi-GPU CAGRA, use host (NumPy) arrays
        >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
        ...     np.float32)
        >>> build_params = cagra.IndexParams(metric="sqeuclidean")
        >>> index = cagra.build(build_params, dataset)
        >>> distances, neighbors = cagra.search(cagra.SearchParams(),
        ...                                         index, dataset, k)
        >>> # Results are already in host memory (NumPy arrays)
        
    """
def distribute(*args, resources = None, **kwargs):
    """
    distribute(filename, resources=None)
    
        Distribute a single-GPU CAGRA index across multiple GPUs from a file.
    
        Parameters
        ----------
        filename : str
            The filename to distribute the index from.
        resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.
            If Multi-GPU Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index : Index
            The distributed index.
    
        Examples
        --------
    
        >>> from cuvs.neighbors.mg import cagra
        >>> index = cagra.distribute("single_gpu_index.bin")  # doctest: +SKIP
        
    """
def extend(*args, resources = None, **kwargs):
    """
    extend(Index index, new_vectors, new_indices=None, resources=None)
    
        Extend the multi-GPU CAGRA index with new vectors.
    
        Parameters
        ----------
        index : :py:class:`cuvs.neighbors.cagra.Index`
        new_vectors : Array interface compliant matrix shape (n_new_vectors, dim)
            Supported dtype [float32, float16, int8, uint8]
            **IMPORTANT**: For multi-GPU CAGRA, new_vectors MUST be in host
            memory (CPU). If using CuPy/device arrays, transfer to host with
            array.get() or cp.asnumpy(array).
        new_indices : Array interface compliant matrix shape (n_new_vectors,),
                      optional
            If provided, these indices will be used for the new vectors.
            If not provided, indices will be automatically assigned.
            **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.
            Expected dtype: uint32
        resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.
            If Multi-GPU Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
    
        >>> import numpy as np
        >>> from cuvs.neighbors.mg import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_new_vectors = 1000
        >>> # For multi-GPU CAGRA, use host (NumPy) arrays
        >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
        ...     np.float32)
        >>> new_vectors = np.random.random_sample(
        ...     (n_new_vectors, n_features)).astype(np.float32)
        >>> new_indices = np.arange(n_samples, n_samples + n_new_vectors,
        ...                         dtype=np.uint32)
        >>> build_params = cagra.IndexParams(metric="sqeuclidean")
        >>> index = cagra.build(build_params, dataset)
        >>> cagra.extend(index, new_vectors, new_indices)  # doctest: +SKIP
        
    """
def load(*args, resources = None, **kwargs):
    """
    load(filename, resources=None)
    
        Deserialize the multi-GPU CAGRA index from a file.
    
        Parameters
        ----------
        filename : str
            The filename to deserialize the index from.
        resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.
            If Multi-GPU Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index : Index
            The deserialized index.
    
        Examples
        --------
    
        >>> from cuvs.neighbors.mg import cagra
        >>> index = cagra.load("index.bin")  # doctest: +SKIP
        
    """
def save(*args, resources = None, **kwargs):
    """
    save(Index index, filename, resources=None)
    
        Serialize the multi-GPU CAGRA index to a file.
    
        Parameters
        ----------
        index : :py:class:`cuvs.neighbors.cagra.Index`
        filename : str
            The filename to serialize the index to.
        resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.
            If Multi-GPU Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
    
        >>> import numpy as np
        >>> from cuvs.neighbors.mg import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> # For multi-GPU CAGRA, use host (NumPy) arrays
        >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
        ...     np.float32)
        >>> build_params = cagra.IndexParams(metric="sqeuclidean")
        >>> index = cagra.build(build_params, dataset)
        >>> cagra.save(index, "index.bin")
        
    """
def search(*args, resources = None, **kwargs):
    """
    search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None)
    
        Search the multi-GPU CAGRA index for the k-nearest neighbors of each query.
    
        Parameters
        ----------
        search_params : :py:class:`cuvs.neighbors.cagra.SearchParams`
        index : :py:class:`cuvs.neighbors.cagra.Index`
        queries : Array interface compliant matrix shape (n_queries, dim)
            Supported dtype [float32, float16, int8, uint8]
            **IMPORTANT**: For multi-GPU CAGRA, queries MUST be in host memory
            (CPU). If using CuPy/device arrays, transfer to host with
            array.get() or cp.asnumpy(array).
        k : int
            The number of neighbors to search for each query.
        neighbors : Array interface compliant matrix shape (n_queries, k), optional
            If provided, this array will be filled with the indices of
            the k-nearest neighbors.
            If not provided, a new host array will be allocated.
            **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.
            Expected dtype: int64
        distances : Array interface compliant matrix shape (n_queries, k), optional
            If provided, this array will be filled with the distances
            to the k-nearest neighbors.
            If not provided, a new host array will be allocated.
            **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.
        resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.
            If Multi-GPU Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        distances : numpy.ndarray
            The distances to the k-nearest neighbors for each query
            (in host memory).
        neighbors : numpy.ndarray
            The indices of the k-nearest neighbors for each query
            (in host memory).
    
        Examples
        --------
    
        >>> import numpy as np
        >>> from cuvs.neighbors.mg import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> # For multi-GPU CAGRA, use host (NumPy) arrays
        >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
        ...     np.float32)
        >>> queries = np.random.random_sample((n_queries, n_features)).astype(
        ...     np.float32)
        >>> build_params = cagra.IndexParams(metric="sqeuclidean")
        >>> index = cagra.build(build_params, dataset)
        >>> distances, neighbors = cagra.search(cagra.SearchParams(),
        ...                                         index, queries, k)
        >>> # Results are already in host memory (NumPy arrays)
        
    """
__test__: dict = {'build (line 141)': '\n    Build the multi-GPU CAGRA index from the dataset for efficient search.\n\n    Parameters\n    ----------\n    index_params : :py:class:`cuvs.neighbors.cagra.IndexParams`\n    dataset : Array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, float16, int8, uint8]\n        **IMPORTANT**: For multi-GPU CAGRA, the dataset MUST be in host\n        memory (CPU). If using CuPy/device arrays, transfer to host with\n        array.get() or cp.asnumpy(array).\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.cagra.Index`\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from cuvs.neighbors.mg import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> # For multi-GPU CAGRA, use host (NumPy) arrays\n    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(\n    ...     np.float32)\n    >>> build_params = cagra.IndexParams(metric="sqeuclidean")\n    >>> index = cagra.build(build_params, dataset)\n    >>> distances, neighbors = cagra.search(cagra.SearchParams(),\n    ...                                         index, dataset, k)\n    >>> # Results are already in host memory (NumPy arrays)\n    ', 'search (line 283)': '\n    Search the multi-GPU CAGRA index for the k-nearest neighbors of each query.\n\n    Parameters\n    ----------\n    search_params : :py:class:`cuvs.neighbors.cagra.SearchParams`\n    index : :py:class:`cuvs.neighbors.cagra.Index`\n    queries : Array interface compliant matrix shape (n_queries, dim)\n        Supported dtype [float32, float16, int8, uint8]\n        **IMPORTANT**: For multi-GPU CAGRA, queries MUST be in host memory\n        (CPU). If using CuPy/device arrays, transfer to host with\n        array.get() or cp.asnumpy(array).\n    k : int\n        The number of neighbors to search for each query.\n    neighbors : Array interface compliant matrix shape (n_queries, k), optional\n        If provided, this array will be filled with the indices of\n        the k-nearest neighbors.\n        If not provided, a new host array will be allocated.\n        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.\n        Expected dtype: int64\n    distances : Array interface compliant matrix shape (n_queries, k), optional\n        If provided, this array will be filled with the distances\n        to the k-nearest neighbors.\n        If not provided, a new host array will be allocated.\n        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.\n    {resources_docstring}\n\n    Returns\n    -------\n    distances : numpy.ndarray\n        The distances to the k-nearest neighbors for each query\n        (in host memory).\n    neighbors : numpy.ndarray\n        The indices of the k-nearest neighbors for each query\n        (in host memory).\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from cuvs.neighbors.mg import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> # For multi-GPU CAGRA, use host (NumPy) arrays\n    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(\n    ...     np.float32)\n    >>> queries = np.random.random_sample((n_queries, n_features)).astype(\n    ...     np.float32)\n    >>> build_params = cagra.IndexParams(metric="sqeuclidean")\n    >>> index = cagra.build(build_params, dataset)\n    >>> distances, neighbors = cagra.search(cagra.SearchParams(),\n    ...                                         index, queries, k)\n    >>> # Results are already in host memory (NumPy arrays)\n    ', 'extend (line 395)': '\n    Extend the multi-GPU CAGRA index with new vectors.\n\n    Parameters\n    ----------\n    index : :py:class:`cuvs.neighbors.cagra.Index`\n    new_vectors : Array interface compliant matrix shape (n_new_vectors, dim)\n        Supported dtype [float32, float16, int8, uint8]\n        **IMPORTANT**: For multi-GPU CAGRA, new_vectors MUST be in host\n        memory (CPU). If using CuPy/device arrays, transfer to host with\n        array.get() or cp.asnumpy(array).\n    new_indices : Array interface compliant matrix shape (n_new_vectors,),\n                  optional\n        If provided, these indices will be used for the new vectors.\n        If not provided, indices will be automatically assigned.\n        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.\n        Expected dtype: uint32\n    {resources_docstring}\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from cuvs.neighbors.mg import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_new_vectors = 1000\n    >>> # For multi-GPU CAGRA, use host (NumPy) arrays\n    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(\n    ...     np.float32)\n    >>> new_vectors = np.random.random_sample(\n    ...     (n_new_vectors, n_features)).astype(np.float32)\n    >>> new_indices = np.arange(n_samples, n_samples + n_new_vectors,\n    ...                         dtype=np.uint32)\n    >>> build_params = cagra.IndexParams(metric="sqeuclidean")\n    >>> index = cagra.build(build_params, dataset)\n    >>> cagra.extend(index, new_vectors, new_indices)  # doctest: +SKIP\n    ', 'save (line 468)': '\n    Serialize the multi-GPU CAGRA index to a file.\n\n    Parameters\n    ----------\n    index : :py:class:`cuvs.neighbors.cagra.Index`\n    filename : str\n        The filename to serialize the index to.\n    {resources_docstring}\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from cuvs.neighbors.mg import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> # For multi-GPU CAGRA, use host (NumPy) arrays\n    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(\n    ...     np.float32)\n    >>> build_params = cagra.IndexParams(metric="sqeuclidean")\n    >>> index = cagra.build(build_params, dataset)\n    >>> cagra.save(index, "index.bin")\n    ', 'load (line 506)': '\n    Deserialize the multi-GPU CAGRA index from a file.\n\n    Parameters\n    ----------\n    filename : str\n        The filename to deserialize the index from.\n    {resources_docstring}\n\n    Returns\n    -------\n    index : Index\n        The deserialized index.\n\n    Examples\n    --------\n\n    >>> from cuvs.neighbors.mg import cagra\n    >>> index = cagra.load("index.bin")  # doctest: +SKIP\n    ', 'distribute (line 540)': '\n    Distribute a single-GPU CAGRA index across multiple GPUs from a file.\n\n    Parameters\n    ----------\n    filename : str\n        The filename to distribute the index from.\n    {resources_docstring}\n\n    Returns\n    -------\n    index : Index\n        The distributed index.\n\n    Examples\n    --------\n\n    >>> from cuvs.neighbors.mg import cagra\n    >>> index = cagra.distribute("single_gpu_index.bin")  # doctest: +SKIP\n    '}
