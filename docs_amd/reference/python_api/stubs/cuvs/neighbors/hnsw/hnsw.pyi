from __future__ import annotations
import builtins as __builtins__
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
import os as os
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
import uuid as uuid
__all__: list[str] = ['DISTANCE_TYPES', 'ExtendParams', 'Index', 'IndexParams', 'SearchParams', 'auto_convert_output', 'auto_sync_resources', 'check_cuvs', 'cuda_interruptible', 'extend', 'from_cagra', 'load', 'np', 'os', 'save', 'search', 'uuid', 'wrap_array']
class ExtendParams:
    """
    ExtendParams(num_threads=0, *)

        Parameters to extend the HNSW index with new data

        Parameters
        ----------
        num_threads : int, default = 0 (optional)
            Number of CPU threads used to increase construction parallelism.
            When set to 0, the number of threads is automatically determined.

    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        ExtendParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        ExtendParams.__setstate_cython__(self, __pyx_state)
        """
class Index:
    """

        HNSW index object. This object stores the trained HNSW index state
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
class IndexParams:
    """
    IndexParams(hierarchy=u'none', *, ef_construction=200, num_threads=0)

        Parameters to build index for HNSW nearest neighbor search

        Parameters
        ----------
        hierarchy : string, default = "none" (optional)
            The hierarchy of the HNSW index. Valid values are ["none", "cpu"].
            - "none": No hierarchy is built.
            - "cpu": Hierarchy is built using CPU.
            - "gpu": Hierarchy is built using GPU.
        ef_construction : int, default = 200 (optional)
            Maximum number of candidate list size used during construction
            when hierarchy is `cpu`.
        num_threads : int, default = 0 (optional)
            Number of CPU threads used to increase construction parallelism
            when hierarchy is `cpu` or `gpu`. When the value is 0, the number of
            threads is automatically determined to the maximum number of threads
            available.
            NOTE: When hierarchy is `gpu`, while the majority of the work is done
            on the GPU, initialization of the HNSW index itself and some other
            work is parallelized with the help of CPU threads.

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
class SearchParams:
    """
    SearchParams(ef=200, *, num_threads=0)

        HNSW search parameters

        Parameters
        ----------
        ef: int, default = 200
            Maximum number of candidate list size used during search.
        num_threads: int, default = 0
            Number of CPU threads used to increase search parallelism.
            When set to 0, the number of threads is automatically determined
            using OpenMP's `omp_get_max_threads()`.

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
    def __repr__(*args, **kwargs):
        ...
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        SearchParams.__setstate_cython__(self, __pyx_state)
        """
def __reduce_cython__(self):
    """
    SearchParams.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    SearchParams.__setstate_cython__(self, __pyx_state)
    """
def extend(*args, resources = None, **kwargs):
    """
    extend(ExtendParams extend_params, Index index, data, resources=None)

        Extends the HNSW index with new data.

        Parameters
        ----------
        extend_params : ExtendParams
        index : Index
            Trained HNSW index.
        data : Host array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, int8, uint8]
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Examples
        --------
        >>> import numpy as np
        >>> from cuvs.neighbors import hnsw, cagra
        >>>
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = np.random.random_sample((n_samples, n_features))
        >>>
        >>> # Build index
        >>> index = cagra.build(hnsw.IndexParams(), dataset)
        >>> # Load index
        >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(hierarchy="cpu"), index)
        >>> # Extend the index with new data
        >>> new_data = np.random.random_sample((n_samples, n_features))
        >>> hnsw.extend(hnsw.ExtendParams(), hnsw_index, new_data)

    """
def from_cagra(*args, resources = None, **kwargs):
    """
    from_cagra(IndexParams index_params, Index cagra_index, temporary_index_path=None, resources=None)

        Returns an HNSW index from a CAGRA index.

        NOTE: When `index_params.hierarchy` is:
              1. `NONE`: This method uses the filesystem to write the CAGRA index
                         in `/tmp/<random_number>.bin` before reading it as an
                         hnswlib index, then deleting the temporary file. The
                         returned index is immutable and can only be searched by
                         the hnswlib wrapper in cuVS, as the format is not
                        compatible with the original hnswlib.
              2. `CPU`: The returned index is mutable and can be extended with
                        additional vectors. The serialized index is also compatible
                        with the original hnswlib library.

        Saving / loading the index is experimental. The serialization format is
        subject to change.

        Parameters
        ----------
        index_params : IndexParams
            Parameters to convert the CAGRA index to HNSW index.
        cagra_index : cagra.Index
            Trained CAGRA index.
        temporary_index_path : string, default = None
            Path to save the temporary index file. If None, the temporary file
            will be saved in `/tmp/<random_number>.bin`.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import cagra
        >>> from cuvs.neighbors import hnsw
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = cagra.build(cagra.IndexParams(), dataset)
        >>> # Serialize the CAGRA index to hnswlib base layer only index format
        >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)

    """
def load(*args, resources = None, **kwargs):
    """
    load(IndexParams index_params, filename, dim, dtype, metric=u'sqeuclidean', resources=None)

        Loads an HNSW index.
        If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,
        then the loaded index is immutable and can only be searched by the hnswlib
        wrapper in cuVS, as the format is not compatible with the original hnswlib.
        However, if the index was constructed with
        `hnsw.IndexParams(hierarchy="cpu")`, then the loaded index is mutable and
        compatible with the original hnswlib.

        Saving / loading the index is experimental. The serialization format is
        subject to change, therefore loading an index saved with a previous
        version of cuVS is not guaranteed to work.

        Parameters
        ----------
        index_params : IndexParams
            Parameters that were used to convert CAGRA index to HNSW index.
        filename : string
            Name of the file.
        dim : int
            Dimensions of the training dataest
        dtype : np.dtype of the saved index
            Valid values for dtype: [np.float32, np.byte, np.ubyte]
        metric : string denoting the metric type, default="sqeuclidean"
            Valid values for metric: ["sqeuclidean", "inner_product"], where
                - sqeuclidean is the euclidean distance without the square root
                  operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
                - inner_product distance is defined as
                  distance(a, b) = \\sum_i a_i * b_i.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Returns
        -------
        index : HnswIndex

        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import cagra
        >>> from cuvs.neighbors import hnsw
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = cagra.build(cagra.IndexParams(), dataset)
        >>> # Serialize the CAGRA index to hnswlib base layer only index format
        >>> hnsw.save("my_index.bin", index)
        >>> index = hnsw.load("my_index.bin", n_features, np.float32,
        ...                   "sqeuclidean")

    """
def save(*args, resources = None, **kwargs):
    """
    save(filename, Index index, resources=None)

        Saves the CAGRA index to a file as an hnswlib index.
        If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,
        then the saved index is immutable and can only be searched by the hnswlib
        wrapper in cuVS, as the format is not compatible with the original hnswlib.
        However, if the index was constructed with
        `hnsw.IndexParams(hierarchy="cpu")`, then the saved index is mutable and
        compatible with the original hnswlib.

        Saving / loading the index is experimental. The serialization format is
        subject to change.

        Parameters
        ----------
        filename : string
            Name of the file.
        index : Index
            Trained HNSW index.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> cagra_index = cagra.build(cagra.IndexParams(), dataset)
        >>> # Serialize and deserialize the cagra index built
        >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), cagra_index)
        >>> hnsw.save("my_index.bin", hnsw_index)

    """
def search(*args, resources = None, **kwargs):
    """
    search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None)

        Find the k nearest neighbors for each query.

        Parameters
        ----------
        search_params : SearchParams
        index : Index
            Trained HNSW index.
        queries : CPU array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float, int]
        k : int
            The number of neighbors.
        neighbors : Optional CPU array interface compliant matrix shape
                    (n_queries, k), dtype uint64_t. If supplied, neighbor
                    indices will be written here in-place. (default None)
        distances : Optional CPU array interface compliant matrix shape
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
        >>> from cuvs.neighbors import cagra, hnsw
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = cagra.build(cagra.IndexParams(), dataset)
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> k = 10
        >>> search_params = hnsw.SearchParams(
        ...     ef=200,
        ...     num_threads=0
        ... )
        >>> # Convert CAGRA index to HNSW
        >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)
        >>> # Using a pooling allocator reduces overhead of temporary array
        >>> # creation during search. This is useful if multiple searches
        >>> # are performed with same query size.
        >>> distances, neighbors = hnsw.search(search_params, index, queries,
        ...                                     k)
        >>> neighbors = cp.asarray(neighbors)
        >>> distances = cp.asarray(distances)

    """
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'save (line 163)': '\n    Saves the CAGRA index to a file as an hnswlib index.\n    If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,\n    then the saved index is immutable and can only be searched by the hnswlib\n    wrapper in cuVS, as the format is not compatible with the original hnswlib.\n    However, if the index was constructed with\n    `hnsw.IndexParams(hierarchy="cpu")`, then the saved index is mutable and\n    compatible with the original hnswlib.\n\n    Saving / loading the index is experimental. The serialization format is\n    subject to change.\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    index : Index\n        Trained HNSW index.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> cagra_index = cagra.build(cagra.IndexParams(), dataset)\n    >>> # Serialize and deserialize the cagra index built\n    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), cagra_index)\n    >>> hnsw.save("my_index.bin", hnsw_index)\n    ', 'load (line 206)': '\n    Loads an HNSW index.\n    If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,\n    then the loaded index is immutable and can only be searched by the hnswlib\n    wrapper in cuVS, as the format is not compatible with the original hnswlib.\n    However, if the index was constructed with\n    `hnsw.IndexParams(hierarchy="cpu")`, then the loaded index is mutable and\n    compatible with the original hnswlib.\n\n    Saving / loading the index is experimental. The serialization format is\n    subject to change, therefore loading an index saved with a previous\n    version of cuVS is not guaranteed to work.\n\n    Parameters\n    ----------\n    index_params : IndexParams\n        Parameters that were used to convert CAGRA index to HNSW index.\n    filename : string\n        Name of the file.\n    dim : int\n        Dimensions of the training dataest\n    dtype : np.dtype of the saved index\n        Valid values for dtype: [np.float32, np.byte, np.ubyte]\n    metric : string denoting the metric type, default="sqeuclidean"\n        Valid values for metric: ["sqeuclidean", "inner_product"], where\n            - sqeuclidean is the euclidean distance without the square root\n              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,\n            - inner_product distance is defined as\n              distance(a, b) = \\sum_i a_i * b_i.\n    {resources_docstring}\n\n    Returns\n    -------\n    index : HnswIndex\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra\n    >>> from cuvs.neighbors import hnsw\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = cagra.build(cagra.IndexParams(), dataset)\n    >>> # Serialize the CAGRA index to hnswlib base layer only index format\n    >>> hnsw.save("my_index.bin", index)\n    >>> index = hnsw.load("my_index.bin", n_features, np.float32,\n    ...                   "sqeuclidean")\n    ', 'from_cagra (line 294)': '\n    Returns an HNSW index from a CAGRA index.\n\n    NOTE: When `index_params.hierarchy` is:\n          1. `NONE`: This method uses the filesystem to write the CAGRA index\n                     in `/tmp/<random_number>.bin` before reading it as an\n                     hnswlib index, then deleting the temporary file. The\n                     returned index is immutable and can only be searched by\n                     the hnswlib wrapper in cuVS, as the format is not\n                    compatible with the original hnswlib.\n          2. `CPU`: The returned index is mutable and can be extended with\n                    additional vectors. The serialized index is also compatible\n                    with the original hnswlib library.\n\n    Saving / loading the index is experimental. The serialization format is\n    subject to change.\n\n    Parameters\n    ----------\n    index_params : IndexParams\n        Parameters to convert the CAGRA index to HNSW index.\n    cagra_index : cagra.Index\n        Trained CAGRA index.\n    temporary_index_path : string, default = None\n        Path to save the temporary index file. If None, the temporary file\n        will be saved in `/tmp/<random_number>.bin`.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra\n    >>> from cuvs.neighbors import hnsw\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = cagra.build(cagra.IndexParams(), dataset)\n    >>> # Serialize the CAGRA index to hnswlib base layer only index format\n    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)\n    ', 'extend (line 353)': '\n    Extends the HNSW index with new data.\n\n    Parameters\n    ----------\n    extend_params : ExtendParams\n    index : Index\n        Trained HNSW index.\n    data : Host array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, int8, uint8]\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from cuvs.neighbors import hnsw, cagra\n    >>>\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = np.random.random_sample((n_samples, n_features))\n    >>>\n    >>> # Build index\n    >>> index = cagra.build(hnsw.IndexParams(), dataset)\n    >>> # Load index\n    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(hierarchy="cpu"), index)\n    >>> # Extend the index with new data\n    >>> new_data = np.random.random_sample((n_samples, n_features))\n    >>> hnsw.extend(hnsw.ExtendParams(), hnsw_index, new_data)\n    ', 'search (line 438)': '\n    Find the k nearest neighbors for each query.\n\n    Parameters\n    ----------\n    search_params : SearchParams\n    index : Index\n        Trained HNSW index.\n    queries : CPU array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float, int]\n    k : int\n        The number of neighbors.\n    neighbors : Optional CPU array interface compliant matrix shape\n                (n_queries, k), dtype uint64_t. If supplied, neighbor\n                indices will be written here in-place. (default None)\n    distances : Optional CPU array interface compliant matrix shape\n                (n_queries, k) If supplied, the distances to the\n                neighbors will be written here in-place. (default None)\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra, hnsw\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = cagra.build(cagra.IndexParams(), dataset)\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 10\n    >>> search_params = hnsw.SearchParams(\n    ...     ef=200,\n    ...     num_threads=0\n    ... )\n    >>> # Convert CAGRA index to HNSW\n    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)\n    >>> # Using a pooling allocator reduces overhead of temporary array\n    >>> # creation during search. This is useful if multiple searches\n    >>> # are performed with same query size.\n    >>> distances, neighbors = hnsw.search(search_params, index, queries,\n    ...                                     k)\n    >>> neighbors = cp.asarray(neighbors)\n    >>> distances = cp.asarray(distances)\n    '}
