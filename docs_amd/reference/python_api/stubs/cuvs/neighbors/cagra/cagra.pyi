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
import warnings as warnings
__all__: list[str] = ['CompressionParams', 'DISTANCE_TYPES', 'Index', 'IndexParams', 'SearchParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'build_index', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'load', 'no_filter', 'np', 'save', 'search', 'warnings', 'wrap_array']
class CompressionParams:
    """
    CompressionParams(pq_bits=8, *, pq_dim=0, vq_n_centers=0, kmeans_n_iters=25, vq_kmeans_trainset_fraction=0.0, pq_kmeans_trainset_fraction=0.0)

        Parameters for VPQ Compression

        Parameters
        ----------
        pq_bits: int
            The bit length of the vector element after compression by PQ.
            Possible values: [4, 5, 6, 7, 8]. The smaller the 'pq_bits', the
            smaller the index size and the better the search performance, but
            the lower the recall.
        pq_dim: int
            The dimensionality of the vector after compression by PQ. When zero,
            an optimal value is selected using a heuristic.
        vq_n_centers: int
            Vector Quantization (VQ) codebook size - number of "coarse cluster
            centers". When zero, an optimal value is selected using a heuristic.
        kmeans_n_iters: int
            The number of iterations searching for kmeans centers (both VQ & PQ
            phases).
        vq_kmeans_trainset_fraction: float
            The fraction of data to use during iterative kmeans building (VQ
            phase). When zero, an optimal value is selected using a heuristic.
        vq_kmeans_trainset_fraction: float
            The fraction of data to use during iterative kmeans building (PQ
            phase). When zero, an optimal value is selected using a heuristic.

    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        CompressionParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        CompressionParams.__setstate_cython__(self, __pyx_state)
        """
    def get_handle(self):
        """
        CompressionParams.get_handle(self)
        """
class Index:
    """

        CAGRA index object. This object stores the trained CAGRA index state
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
    IndexParams(metric=u'sqeuclidean', *, intermediate_graph_degree=128, graph_degree=64, build_algo=u'ivf_pq', nn_descent_niter=20, compression=None)

        Parameters to build index for CAGRA nearest neighbor search

        Parameters
        ----------
        metric : string denoting the metric type, default="sqeuclidean"
            Valid values for metric: ["sqeuclidean", "inner_product"], where
                - sqeuclidean is the euclidean distance without the square root
                  operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2
                - inner_product distance is defined as
                  distance(a, b) = \\sum_i a_i * b_i.
        intermediate_graph_degree : int, default = 128

        graph_degree : int, default = 64

        build_algo: string denoting the graph building algorithm to use,                 default = "ivf_pq"
            Valid values for algo: ["ivf_pq", "nn_descent",
                                    "iterative_cagra_search"], where
                - ivf_pq will use the IVF-PQ algorithm for building the knn graph
                - nn_descent (experimental) will use the NN-Descent algorithm for
                  building the knn graph. It is expected to be generally
                  faster than ivf_pq.
                - iterative_cagra_search will iteratively build the knn graph using
                  CAGRA's search() and optimize()
        compression: CompressionParams, optional
            If compression is desired should be a CompressionParams object. If None
            compression will be disabled.

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
    SearchParams(max_queries=0, *, itopk_size=64, max_iterations=0, algo=u'auto', team_size=0, search_width=1, min_iterations=0, thread_block_size=0, hashmap_mode=u'auto', hashmap_min_bitlen=0, hashmap_max_fill_rate=0.5, num_random_samplings=1, rand_xor_mask=0x128394)

        CAGRA search parameters

        Parameters
        ----------
        max_queries: int, default = 0
            Maximum number of queries to search at the same time (batch size).
            Auto select when 0.
        itopk_size: int, default = 64
            Number of intermediate search results retained during the search.
            This is the main knob to adjust trade off between accuracy and
            search speed. Higher values improve the search accuracy.
        max_iterations: int, default = 0
            Upper limit of search iterations. Auto select when 0.
        algo: string denoting the search algorithm to use, default = "auto"
            Valid values for algo: ["auto", "single_cta", "multi_cta"], where
                - auto will automatically select the best value based on query size
                - single_cta is better when query contains larger number of
                  vectors (e.g >10)
                - multi_cta is better when query contains only a few vectors
        team_size: int, default = 0
            Number of threads used to calculate a single distance. 4, 8, 16,
            or 32.
        search_width: int, default = 1
            Number of graph nodes to select as the starting point for the
            search in each iteration.
        min_iterations: int, default = 0
            Lower limit of search iterations.
        thread_block_size: int, default = 0
            Thread block size. 0, 64, 128, 256, 512, 1024.
            Auto selection when 0.
        hashmap_mode: string denoting the type of hash map to use.
            It's usually better to allow the algorithm to select this value,
            default = "auto".
            Valid values for hashmap_mode: ["auto", "small", "hash"], where
                - auto will automatically select the best value based on algo
                - small will use the small shared memory hash table with resetting.
                - hash will use a single hash table in global memory.
        hashmap_min_bitlen: int, default = 0
            Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
        hashmap_max_fill_rate: float, default = 0.5
            Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
        num_random_samplings: int, default = 1
            Number of iterations of initial random seed node selection. 1 or
            more.
        rand_xor_mask: int, default = 0x128394
            Bit mask used for initial random seed node selection.

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
def build(*args, resources = None, **kwargs):
    """
    build(IndexParams index_params, dataset, resources=None)

        Build the CAGRA index from the dataset for efficient search.

        The build performs two different steps- first an intermediate knn-graph is
        constructed, then it's optimized it to create the final graph. The
        index_params object controls the node degree of these graphs.

        It is required that both the dataset and the optimized graph fit the
        GPU memory.

        The following distance metrics are supported:
            - L2
            - InnerProduct

        Parameters
        ----------
        index_params : IndexParams object
        dataset : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float, int8, uint8]
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Returns
        -------
        index: cuvs.cagra.Index

        Examples
        --------

        >>> import cupy as cp
        >>> from cuvs.neighbors import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> build_params = cagra.IndexParams(metric="sqeuclidean")
        >>> index = cagra.build(build_params, dataset)
        >>> distances, neighbors = cagra.search(cagra.SearchParams(),
        ...                                      index, dataset,
        ...                                      k)
        >>> distances = cp.asarray(distances)
        >>> neighbors = cp.asarray(neighbors)

    """
def build_index(index_params, dataset, resources = None):
    """
    build_index(IndexParams index_params, dataset, resources=None)
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
            Trained CAGRA index.
        include_dataset : bool
            Whether or not to write out the dataset along with the index. Including
            the dataset in the serialized index will use extra disk space, and
            might not be desired if you already have a copy of the dataset on
            disk. If this option is set to false, you will have to call
            `index.update_dataset(dataset)` after loading the index.
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
        >>> index = cagra.build(cagra.IndexParams(), dataset)
        >>> # Serialize and deserialize the cagra index built
        >>> cagra.save("my_index.bin", index)
        >>> index_loaded = cagra.load("my_index.bin")

    """
def search(*args, resources = None, **kwargs):
    """
    search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None, filter=None)

        Find the k nearest neighbors for each query.

        Parameters
        ----------
        search_params : SearchParams
        index : Index
            Trained CAGRA index.
        queries : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float, int8, uint8]
        k : int
            The number of neighbors.
        neighbors : Optional CUDA array interface compliant matrix shape
                    (n_queries, k), dtype int64_t. If supplied, neighbor
                    indices will be written here in-place. (default None)
        distances : Optional CUDA array interface compliant matrix shape
                    (n_queries, k) If supplied, the distances to the
                    neighbors will be written here in-place. (default None)
        filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter
                    neighbors based on a given bitset.
            (default None)
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
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = cagra.build(cagra.IndexParams(), dataset)
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> k = 10
        >>> search_params = cagra.SearchParams(
        ...     max_queries=100,
        ...     itopk_size=64
        ... )
        >>> # Using a pooling allocator reduces overhead of temporary array
        >>> # creation during search. This is useful if multiple searches
        >>> # are performed with same query size.
        >>> distances, neighbors = cagra.search(search_params, index, queries,
        ...                                     k)
        >>> neighbors = cp.asarray(neighbors)
        >>> distances = cp.asarray(distances)

    """
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 249)': '\n    Build the CAGRA index from the dataset for efficient search.\n\n    The build performs two different steps- first an intermediate knn-graph is\n    constructed, then it\'s optimized it to create the final graph. The\n    index_params object controls the node degree of these graphs.\n\n    It is required that both the dataset and the optimized graph fit the\n    GPU memory.\n\n    The following distance metrics are supported:\n        - L2\n        - InnerProduct\n\n    Parameters\n    ----------\n    index_params : IndexParams object\n    dataset : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float, int8, uint8]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: cuvs.cagra.Index\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = cagra.IndexParams(metric="sqeuclidean")\n    >>> index = cagra.build(build_params, dataset)\n    >>> distances, neighbors = cagra.search(cagra.SearchParams(),\n    ...                                      index, dataset,\n    ...                                      k)\n    >>> distances = cp.asarray(distances)\n    >>> neighbors = cp.asarray(neighbors)\n    ', 'search (line 491)': '\n    Find the k nearest neighbors for each query.\n\n    Parameters\n    ----------\n    search_params : SearchParams\n    index : Index\n        Trained CAGRA index.\n    queries : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float, int8, uint8]\n    k : int\n        The number of neighbors.\n    neighbors : Optional CUDA array interface compliant matrix shape\n                (n_queries, k), dtype int64_t. If supplied, neighbor\n                indices will be written here in-place. (default None)\n    distances : Optional CUDA array interface compliant matrix shape\n                (n_queries, k) If supplied, the distances to the\n                neighbors will be written here in-place. (default None)\n    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter\n                neighbors based on a given bitset.\n        (default None)\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = cagra.build(cagra.IndexParams(), dataset)\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 10\n    >>> search_params = cagra.SearchParams(\n    ...     max_queries=100,\n    ...     itopk_size=64\n    ... )\n    >>> # Using a pooling allocator reduces overhead of temporary array\n    >>> # creation during search. This is useful if multiple searches\n    >>> # are performed with same query size.\n    >>> distances, neighbors = cagra.search(search_params, index, queries,\n    ...                                     k)\n    >>> neighbors = cp.asarray(neighbors)\n    >>> distances = cp.asarray(distances)\n    ', 'save (line 604)': '\n    Saves the index to a file.\n\n    Saving / loading the index is experimental. The serialization format is\n    subject to change.\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    index : Index\n        Trained CAGRA index.\n    include_dataset : bool\n        Whether or not to write out the dataset along with the index. Including\n        the dataset in the serialized index will use extra disk space, and\n        might not be desired if you already have a copy of the dataset on\n        disk. If this option is set to false, you will have to call\n        `index.update_dataset(dataset)` after loading the index.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = cagra.build(cagra.IndexParams(), dataset)\n    >>> # Serialize and deserialize the cagra index built\n    >>> cagra.save("my_index.bin", index)\n    >>> index_loaded = cagra.load("my_index.bin")\n    '}
