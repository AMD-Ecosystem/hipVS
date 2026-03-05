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
    @staticmethod
    def _get_distances(*args, resources = None, **kwargs):
        """
        Index._get_distances(self, resources=None)
        """
    @staticmethod
    def _get_graph(*args, resources = None, **kwargs):
        """
        Index._get_graph(self, resources=None)
        """
class IndexParams:
    """
    IndexParams(metric=None, *, metric_arg=None, graph_degree=None, intermediate_graph_degree=None, max_iterations=None, termination_threshold=None, return_distances=None)
    
        Parameters to build NN-Descent Index
    
        Parameters
        ----------
        metric : str, default = "sqeuclidean"
            String denoting the metric type.
            Supported metrics are `l2`, `euclidean`, `sqeuclidean`,
            `inner_product`, `cosine`, and `bitwise_hamming`
            (`bitwise_hamming` is for int8 and uint8 data types only)
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
        return_distances : bool
            Whether to return distances array
        
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
        Get a pointer to the underlying C object.
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
            Supported dtype [float32, int8, uint8]
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
__test__: dict = {'build (line 213)': '\n    Build KNN graph from the dataset\n\n    Parameters\n    ----------\n    index_params : :py:class:`cuvs.neighbors.nn_descent.IndexParams`\n    dataset : Array interface compliant matrix, on either host or device memory\n        Supported dtype [float32, int8, uint8]\n    graph : Optional host matrix for storing output graph\n    {resources_docstring}\n\n    Returns\n    -------\n    index: py:class:`cuvs.neighbors.nn_descent.Index`\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import nn_descent\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> n_queries = 1000\n    >>> k = 10\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = nn_descent.IndexParams(metric="sqeuclidean")\n    >>> index = nn_descent.build(build_params, dataset)\n    >>> graph = index.graph\n    '}
