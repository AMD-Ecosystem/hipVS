from __future__ import annotations
import builtins as __builtins__
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import cai_wrapper
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['DISTANCE_NAMES', 'DISTANCE_TYPES', 'Index', 'IndexParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'np', 'wrap_array']
class Index:
    """

        NN-Descent index object. This object stores the trained NN-Descent index,
        which can be used to get the NN-Descent graph and distances after
        building

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
    IndexParams(metric=None, *, metric_arg=None, graph_degree=None, intermediate_graph_degree=None, max_iterations=None, termination_threshold=None, n_clusters=None)

        Parameters to build NN-Descent Index

        Parameters
        ----------
        metric : str, default = "sqeuclidean"
            String denoting the metric type.
            distribution of the newly added data.
        graph_degree :  int
            For an input dataset of dimensions (N, D), determines the final
            dimensions of the all-neighbors knn graph which turns out to be of
            dimensions (N, graph_degree)
        intermediate_graph_degree : int
            Internally, nn-descent builds an all-neighbors knn graph of dimensions
            (N, intermediate_graph_degree) before selecting the final
            `graph_degree` neighbors. It's recommended that
            `intermediate_graph_degree` >= 1.5 * graph_degree
        max_iterations : int
            The number of iterations that nn-descent will refine the graph for.
            More iterations produce a better quality graph at cost of performance
        termination_threshold : float
            The delta at which nn-descent will terminate its iterations

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
    build(IndexParams index_params, dataset, graph=None, resources=None)

        Build KNN graph from the dataset

        Parameters
        ----------
        index_params : :py:class:`cuvs.neighbors.nn_descent.IndexParams`
        dataset : Array interface compliant matrix, on either host or device memory
            Supported dtype [float, int8, uint8]
        graph : Optional host matrix for storing output graph
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Returns
        -------
        index: py:class:`cuvs.neighbors.nn_descent.Index`

        Examples
        --------

        >>> import cupy as cp
        >>> from cuvs.neighbors import nn_descent
        >>> n_samples = 50000
        >>> n_features = 50
        >>> n_queries = 1000
        >>> k = 10
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> build_params = nn_descent.IndexParams(metric="sqeuclidean")
        >>> index = nn_descent.build(build_params, dataset)
        >>> graph = index.graph

    """
DISTANCE_NAMES: dict = {1: 'euclidean', 0: 'sqeuclidean', 3: 'cityblock', 6: 'inner_product', 7: 'chebyshev', 8: 'canberra', 2: 'cosine', 9: 'minkowski', 10: 'correlation', 11: 'jaccard', 12: 'hellinger', 14: 'braycurtis', 15: 'jensenshannon', 16: 'hamming', 17: 'kl_divergence', 18: 'russellrao', 19: 'dice', 20: 'bitwise_hamming'}
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 179)': '\n    Build KNN graph from the dataset\n\n    Parameters\n    ----------\n    index_params : :py:class:`cuvs.neighbors.nn_descent.IndexParams`\n    dataset : Array interface compliant matrix, on either host or device memory\n        Supported dtype [float, int8, uint8]\n    graph : Optional host matrix for storing output graph\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.nn_descent.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import nn_descent\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = nn_descent.IndexParams(metric="sqeuclidean")\n    >>> index = nn_descent.build(build_params, dataset)\n    >>> graph = index.graph\n    '}
