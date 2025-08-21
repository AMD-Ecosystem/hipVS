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
__all__: list[str] = ['DISTANCE_TYPES', 'Index', 'IndexParams', 'SearchParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'extend', 'load', 'no_filter', 'np', 'save', 'search', 'wrap_array']
class Index:
    """

        IvfFlat index object. This object stores the trained IvfFlat index state
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
    IndexParams(n_lists=1024, *, metric=u'sqeuclidean', metric_arg=2.0, kmeans_n_iters=20, kmeans_trainset_fraction=0.5, adaptive_centers=False, add_data_on_build=True, conservative_memory_allocation=False)

        Parameters to build index for IvfFlat nearest neighbor search

        Parameters
        ----------
        n_lists : int, default = 1024
            The number of clusters used in the coarse quantizer.
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
        kmeans_n_iters : int, default = 20
            The number of iterations searching for kmeans centers during index
            building.
            The default setting is often fine, but this parameter can be decreased
            to improve training time wih larger trainset fractions (10M+ vectors)
            or increased for smaller trainset fractions (very small number of
            vectors) to improve recall.
        kmeans_trainset_fraction : int, default = 0.5
            If kmeans_trainset_fraction is less than 1, then the dataset is
            subsampled, and only n_samples * kmeans_trainset_fraction rows
            are used for training.
        add_data_on_build : bool, default = True
            After training the coarse and fine quantizers, we will populate
            the index with the dataset if add_data_on_build == True, otherwise
            the index is left empty, and the extend method can be used
            to add new vectors to the index.
        adaptive_centers : bool, default = False
            By default (adaptive_centers = False), the cluster centers are
            trained in `ivf_flat.build`, and and never modified in
            `ivf_flat.extend`. The alternative behavior (adaptive_centers
            = true) is to update the cluster centers for new data when it is
            added. In this case, `index.centers()` are always exactly the
            centroids of the data in the corresponding clusters. The drawback
            of this behavior is that the centroids depend on the order of
            adding new data (through the classification of the added data);
            that is, `index.centers()` "drift" together with the changing
            distribution of the newly added data.

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
    SearchParams(n_probes=20, *)

        Supplemental parameters to search IVF-Flat index

        Parameters
        ----------
        n_probes: int
            The number of clusters to search.

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

        Build the IvfFlat index from the dataset for efficient search.

        Parameters
        ----------
        index_params : :py:class:`cuvs.neighbors.ivf_flat.IndexParams`
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
        index: py:class:`cuvs.neighbors.ivf_flat.Index`

        Examples
        --------

        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_flat
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> build_params = ivf_flat.IndexParams(metric="sqeuclidean")
        >>> index = ivf_flat.build(build_params, dataset)
        >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
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
        index : ivf_flat.Index
            Trained ivf_flat object.
        new_vectors : array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float, int8, uint8]
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
        index: py:class:`cuvs.neighbors.ivf_flat.Index`

        Examples
        --------

        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_flat
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
        >>> n_rows = 100
        >>> more_data = cp.random.random_sample((n_rows, n_features),
        ...                                     dtype=cp.float32)
        >>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)
        >>> index = ivf_flat.extend(index, more_data, indices)
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
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
            Trained IVF-Flat index.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import ivf_flat
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
        >>> # Serialize and deserialize the ivf_flat index built
        >>> ivf_flat.save("my_index.bin", index)
        >>> index_loaded = ivf_flat.load("my_index.bin")

    """
def search(*args, resources = None, **kwargs):
    """
    search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None, filter=None)

        Find the k nearest neighbors for each query.

        Parameters
        ----------
        search_params : py:class:`cuvs.neighbors.ivf_flat.SearchParams`
        index : py:class:`cuvs.neighbors.ivf_flat.Index`
            Trained IvfFlat index.
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
        >>> from cuvs.neighbors import ivf_flat
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build the index
        >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
        >>>
        >>> # Search using the built index
        >>> queries = cp.random.random_sample((n_queries, n_features),
        ...                                   dtype=cp.float32)
        >>> k = 10
        >>> search_params = ivf_flat.SearchParams(n_probes=20)
        >>>
        >>> distances, neighbors = ivf_flat.search(search_params, index, queries,
        ...                                     k)

    """
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 185)': '\n    Build the IvfFlat index from the dataset for efficient search.\n\n    Parameters\n    ----------\n    index_params : :py:class:`cuvs.neighbors.ivf_flat.IndexParams`\n    dataset : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float, int8, uint8]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.ivf_flat.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_flat\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = ivf_flat.IndexParams(metric="sqeuclidean")\n    >>> index = ivf_flat.build(build_params, dataset)\n    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),\n    ...                                        index, dataset,\n    ...                                        k)\n    >>> distances = cp.asarray(distances)\n    >>> neighbors = cp.asarray(neighbors)\n    ', 'search (line 270)': '\n    Find the k nearest neighbors for each query.\n\n    Parameters\n    ----------\n    search_params : py:class:`cuvs.neighbors.ivf_flat.SearchParams`\n    index : py:class:`cuvs.neighbors.ivf_flat.Index`\n        Trained IvfFlat index.\n    queries : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float, int8, uint8]\n    k : int\n        The number of neighbors.\n    neighbors : Optional CUDA array interface compliant matrix shape\n                (n_queries, k), dtype int64_t. If supplied, neighbor\n                indices will be written here in-place. (default None)\n    distances : Optional CUDA array interface compliant matrix shape\n                (n_queries, k) If supplied, the distances to the\n                neighbors will be written here in-place. (default None)\n    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter\n                neighbors based on a given bitset. (default None)\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_flat\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build the index\n    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)\n    >>>\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> k = 10\n    >>> search_params = ivf_flat.SearchParams(n_probes=20)\n    >>>\n    >>> distances, neighbors = ivf_flat.search(search_params, index, queries,\n    ...                                     k)\n    ', 'save (line 373)': '\n    Saves the index to a file.\n\n    Saving / loading the index is experimental. The serialization format is\n    subject to change.\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    index : Index\n        Trained IVF-Flat index.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_flat\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)\n    >>> # Serialize and deserialize the ivf_flat index built\n    >>> ivf_flat.save("my_index.bin", index)\n    >>> index_loaded = ivf_flat.load("my_index.bin")\n    ', 'extend (line 443)': '\n    Extend an existing index with new vectors.\n\n    The input array can be either CUDA array interface compliant matrix or\n    array interface compliant matrix in host memory.\n\n\n    Parameters\n    ----------\n    index : ivf_flat.Index\n        Trained ivf_flat object.\n    new_vectors : array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float, int8, uint8]\n    new_indices : array interface compliant vector shape (n_samples)\n        Supported dtype [int64]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.ivf_flat.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import ivf_flat\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)\n    >>> n_rows = 100\n    >>> more_data = cp.random.random_sample((n_rows, n_features),\n    ...                                     dtype=cp.float32)\n    >>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)\n    >>> index = ivf_flat.extend(index, more_data, indices)\n    >>> # Search using the built index\n    >>> queries = cp.random.random_sample((n_queries, n_features),\n    ...                                   dtype=cp.float32)\n    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),\n    ...                                      index, queries,\n    ...                                      k=10)\n    '}
