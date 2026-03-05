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
from cuvs.common.device_tensor_view import DeviceTensorView
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import cai_wrapper
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['DISTANCE_NAMES', 'DISTANCE_TYPES', 'DeviceTensorView', 'Index', 'IndexParams', 'SearchParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'extend', 'load', 'np', 'save', 'search', 'wrap_array']
class Index:
    """
    
        IvfPq index object. This object stores the trained IvfPq index state
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
    def __len__(self):
        """
        Return len(self).
        """
class IndexParams:
    """
    IndexParams(n_lists=1024, *, metric=u'sqeuclidean', metric_arg=2.0, kmeans_n_iters=20, kmeans_trainset_fraction=0.5, pq_bits=8, pq_dim=0, codebook_kind=u'subspace', force_random_rotation=False, add_data_on_build=True, conservative_memory_allocation=False, max_train_points_per_pq_code=256)
    
        Parameters to build index for IvfPq nearest neighbor search
    
        Parameters
        ----------
        n_lists : int, default = 1024
            The number of clusters used in the coarse quantizer.
        metric : str, default="sqeuclidean"
            String denoting the metric type.
            Valid values for metric: ["sqeuclidean", "inner_product",
            "euclidean", "cosine"],
            where:
    
                - sqeuclidean is the euclidean distance without the square root
                  operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
                - euclidean is the euclidean distance
                - inner product distance is defined as
                  distance(a, b) = \\sum_i a_i * b_i.
                - cosine distance is defined as
                  distance(a, b) = 1 - \\sum_i a_i * b_i / ( ||a||_2 * ||b||_2).
    
        kmeans_n_iters : int, default = 20
            The number of iterations searching for kmeans centers during index
            building.
        kmeans_trainset_fraction : int, default = 0.5
            If kmeans_trainset_fraction is less than 1, then the dataset is
            subsampled, and only n_samples * kmeans_trainset_fraction rows
            are used for training.
        pq_bits : int, default = 8
            The bit length of the vector element after quantization.
        pq_dim : int, default = 0
            The dimensionality of a the vector after product quantization.
            When zero, an optimal value is selected using a heuristic. Note
            pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
            results in a smaller index size and better search performance, but
            lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
            but multiple of 8 are desirable for good performance. If 'pq_bits'
            is not 8, 'pq_dim' should be a multiple of 8. For good performance,
            it is desirable that 'pq_dim' is a multiple of 32. Ideally,
            'pq_dim' should be also a divisor of the dataset dim.
        codebook_kind : string, default = "subspace"
            Valid values ["subspace", "cluster"]
        force_random_rotation : bool, default = False
            Apply a random rotation matrix on the input data and queries even
            if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
            a random rotation is always applied to the input data and queries
            to transform the working space from `dim` to `rot_dim`, which may
            be slightly larger than the original space and and is a multiple
            of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
            not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
            hence no need in adding "extra" data columns / features). By
            default, if `dim == rot_dim`, the rotation transform is
            initialized with the identity matrix. When
            `force_random_rotation == True`, a random orthogonal transform
            matrix is generated regardless of the values of `dim` and `pq_dim`.
        add_data_on_build : bool, default = True
            After training the coarse and fine quantizers, we will populate
            the index with the dataset if add_data_on_build == True, otherwise
            the index is left empty, and the extend method can be used
            to add new vectors to the index.
        conservative_memory_allocation : bool, default = True
            By default, the algorithm allocates more space than necessary for
            individual clusters (`list_data`). This allows to amortize the cost
            of memory allocation and reduce the number of data copies during
            repeated calls to `extend` (extending the database).
            To disable this behavior and use as little GPU memory for the
            database as possible, set this flat to `True`.
        max_train_points_per_pq_code : int, default = 256
            The max number of data points to use per PQ code during PQ codebook
            training. Using more data points per PQ code may increase the
            quality of PQ codebook but may also increase the build time. The
            parameter is applied to both PQ codebook generation methods, i.e.,
            PER_SUBSPACE and PER_CLUSTER. In both cases, we will use
            pq_book_size * max_train_points_per_pq_code training points to
            train each codebook.
        
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
class SearchParams:
    """
    SearchParams(n_probes=20, *, lut_dtype=np.float32, internal_distance_dtype=np.float32, coarse_search_dtype=np.float32, max_internal_batch_size=4096)
    
        Supplemental parameters to search IVF-Pq index
    
        Parameters
        ----------
        n_probes: int
            The number of clusters to search.
        lut_dtype: default = np.float32
            Data type of look up table to be created dynamically at search
            time. The use of low-precision types reduces the amount of shared
            memory required at search time, so fast shared memory kernels can
            be used even for datasets with large dimansionality. Note that
            the recall is slightly degraded when low-precision type is
            selected. Possible values [np.float32, np.float16, np.uint8]
        internal_distance_dtype: default = np.float32
            Storage data type for distance/similarity computation.
            Possible values [np.float32, np.float16]
        coarse_search_dtype: default = np.float32
            [Experimental] The data type to use as the GEMM element type when
            searching the clusters to probe.
            Possible values: [np.float32, np.float16, np.int8].
            - Legacy default: np.float32
            - Recommended for performance: np.float16 (half)
            - Experimental/low-precision: np.int8
        max_internal_batch_size: default = 4096
            Set the internal batch size to improve GPU utilization at the cost
            of larger memory footprint.
        
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
    
        Build the IvfPq index from the dataset for efficient search.
    
        The input dataset array can be either CUDA array interface compliant matrix
        or an array interface compliant matrix in host memory.
    
        Parameters
        ----------
        index_params : :py:class:`cuvs.neighbors.ivf_pq.IndexParams`
            Parameters on how to build the index
        dataset : Array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, float16, int8, uint8]
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index: :py:class:`cuvs.neighbors.ivf_pq.Index`
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_pq
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> build_params = ivf_pq.IndexParams(metric="sqeuclidean")
        >>> index = ivf_pq.build(build_params, dataset)
        >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
        ...                                        index, dataset,
        ...                                        k)
        >>> distances = cp.asarray(distances)
        >>> neighbors = cp.asarray(neighbors)
        
    """
def extend(*args, resources = None, **kwargs):
    """
    extend(Index index, new_vectors, new_indices, resources=None)
    
        Extend an existing index with new vectors.
    
        The input array can be either CUDA array interface compliant matrix or
        array interface compliant matrix in host memory.
    
    
        Parameters
        ----------
        index : ivf_pq.Index
            Trained ivf_pq object.
        new_vectors : array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, int8, uint8]
        new_indices : array interface compliant vector shape (n_samples)
            Supported dtype [int64]
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index: py:class:`cuvs.neighbors.ivf_pq.Index`
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_pq
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
        >>> n_rows = 100
        >>> more_data = cp.random.random_sample((n_rows, n_features),
        ...                                     dtype=cp.float32)
        >>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)
        >>> index = ivf_pq.extend(index, more_data, indices)
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
        ...                                      index, queries,
        ...                                      k=10)
        
    """
def load(*args, resources = None, **kwargs):
    """
    load(filename, resources=None)
    
        Loads index from file.
    
        Saving / loading the index is experimental. The serialization format is
        subject to change, therefore loading an index saved with a previous
        version of cuvs is not guaranteed to work.
    
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
    
        
    """
def save(*args, resources = None, **kwargs):
    """
    save(filename, Index index, bool include_dataset=True, resources=None)
    
        Saves the index to a file.
    
        Saving / loading the index is experimental. The serialization format is
        subject to change.
    
        Parameters
        ----------
        filename : string
            Name of the file.
        index : Index
            Trained IVF-PQ index.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_pq
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
        >>> # Serialize and deserialize the ivf_pq index built
        >>> ivf_pq.save("my_index.bin", index)
        >>> index_loaded = ivf_pq.load("my_index.bin")
        
    """
def search(*args, resources = None, **kwargs):
    """
    search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None)
    
        Find the k nearest neighbors for each query.
    
        Parameters
        ----------
        search_params : :py:class:`cuvs.neighbors.ivf_pq.SearchParams`
            Parameters on how to search the index
        index : :py:class:`cuvs.neighbors.ivf_pq.Index`
            Trained IvfPq index.
        queries : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, int8, uint8]
        k : int
            The number of neighbors.
        neighbors : Optional CUDA array interface compliant matrix shape
                    (n_queries, k), dtype int64_t. If supplied, neighbor
                    indices will be written here in-place. (default None)
        distances : Optional CUDA array interface compliant matrix shape
                    (n_queries, k) If supplied, the distances to the
                    neighbors will be written here in-place. (default None)
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_pq
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build the index
        >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
        >>>
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> k = 10
        >>> search_params = ivf_pq.SearchParams(n_probes=20)
        >>>
        >>> distances, neighbors = ivf_pq.search(search_params, index, queries,
        ...                                     k)
        
    """
DISTANCE_NAMES: dict = {1: 'euclidean', 0: 'sqeuclidean', 3: 'cityblock', 6: 'inner_product', 7: 'chebyshev', 8: 'canberra', 2: 'cosine', 9: 'minkowski', 10: 'correlation', 11: 'jaccard', 12: 'hellinger', 14: 'braycurtis', 15: 'jensenshannon', 16: 'hamming', 17: 'kl_divergence', 18: 'russellrao', 19: 'dice', 20: 'bitwise_hamming'}
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 302)': '\n    Build the IvfPq index from the dataset for efficient search.\n\n    The input dataset array can be either CUDA array interface compliant matrix\n    or an array interface compliant matrix in host memory.\n\n    Parameters\n    ----------\n    index_params : :py:class:`cuvs.neighbors.ivf_pq.IndexParams`\n        Parameters on how to build the index\n    dataset : Array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, float16, int8, uint8]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: :py:class:`cuvs.neighbors.ivf_pq.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_pq\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = ivf_pq.IndexParams(metric="sqeuclidean")\n    >>> index = ivf_pq.build(build_params, dataset)\n    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),\n    ...                                        index, dataset,\n    ...                                        k)\n    >>> distances = cp.asarray(distances)\n    >>> neighbors = cp.asarray(neighbors)\n    ', 'search (line 453)': '\n    Find the k nearest neighbors for each query.\n\n    Parameters\n    ----------\n    search_params : :py:class:`cuvs.neighbors.ivf_pq.SearchParams`\n        Parameters on how to search the index\n    index : :py:class:`cuvs.neighbors.ivf_pq.Index`\n        Trained IvfPq index.\n    queries : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, int8, uint8]\n    k : int\n        The number of neighbors.\n    neighbors : Optional CUDA array interface compliant matrix shape\n                (n_queries, k), dtype int64_t. If supplied, neighbor\n                indices will be written here in-place. (default None)\n    distances : Optional CUDA array interface compliant matrix shape\n                (n_queries, k) If supplied, the distances to the\n                neighbors will be written here in-place. (default None)\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_pq\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build the index\n    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)\n    >>>\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 10\n    >>> search_params = ivf_pq.SearchParams(n_probes=20)\n    >>>\n    >>> distances, neighbors = ivf_pq.search(search_params, index, queries,\n    ...                                     k)\n    ', 'save (line 552)': '\n    Saves the index to a file.\n\n    Saving / loading the index is experimental. The serialization format is\n    subject to change.\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    index : Index\n        Trained IVF-PQ index.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_pq\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)\n    >>> # Serialize and deserialize the ivf_pq index built\n    >>> ivf_pq.save("my_index.bin", index)\n    >>> index_loaded = ivf_pq.load("my_index.bin")\n    ', 'extend (line 622)': '\n    Extend an existing index with new vectors.\n\n    The input array can be either CUDA array interface compliant matrix or\n    array interface compliant matrix in host memory.\n\n\n    Parameters\n    ----------\n    index : ivf_pq.Index\n        Trained ivf_pq object.\n    new_vectors : array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, int8, uint8]\n    new_indices : array interface compliant vector shape (n_samples)\n        Supported dtype [int64]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.ivf_pq.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_pq\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)\n    >>> n_rows = 100\n    >>> more_data = cp.random.random_sample((n_rows, n_features),\n    ...                                     dtype=cp.float32)\n    >>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)\n    >>> index = ivf_pq.extend(index, more_data, indices)\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),\n    ...                                      index, queries,\n    ...                                      k=10)\n    '}
