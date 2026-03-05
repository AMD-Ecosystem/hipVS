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
import warnings as warnings
__all__: list[str] = ['DISTANCE_NAMES', 'DISTANCE_TYPES', 'Index', 'IndexParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'cai_wrapper', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'np', 'save', 'warnings', 'wrap_array']
class Index:
    """
    
        Vamana index object. This object stores the trained Vamana index state
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
    IndexParams(metric=u'sqeuclidean', *, graph_degree=32, visited_size=64, vamana_iters=1, alpha=1.2, max_fraction=0.06, batch_base=2.0, queue_size=127, reverse_batchsize=1000000)
    
        Parameters for building a Vamana index
    
        Parameters
        ----------
        metric : str, default="sqeuclidean"
            String denoting the metric type. Supported metrics include:
            - "sqeuclidean"
            - "l2"
        graph_degree : int, default=32
            Maximum degree of graph; corresponds to the R parameter of
            Vamana algorithm in the literature.
        visited_size : int, default=64
            Maximum number of visited nodes per search during Vamana algorithm.
            Loosely corresponds to the L parameter in the literature.
        vamana_iters : float, default=1
            Number of Vamana vector insertion iterations (each iteration inserts
            all vectors).
        alpha : float, default=1.2
            Alpha for pruning parameter. Used to determine how aggressive the
            pruning will be.
        max_fraction : float, default=0.06
            Maximum fraction of dataset inserted per batch. Larger max batch
            decreases graph quality, but improves speed.
        batch_base : float, default=2.0
            Base of growth rate of batch sizes.
        queue_size : int, default=127
            Size of candidate queue structure - should be (2^x)-1.
        reverse_batchsize : int, default=1000000
            Max batchsize of reverse edge processing (reduces memory footprint).
        
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
    IndexParams.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    IndexParams.__setstate_cython__(self, __pyx_state)
    """
def build(*args, resources = None, **kwargs):
    """
    build(IndexParams index_params, dataset, resources=None)
    
        Build the Vamana index from the dataset for efficient search.
    
        The build utilities the Vamana insertion-based algorithm to create
        the graph. The algorithm starts with an empty graph and iteratively
        inserts batches of nodes. Each batch involves performing a greedy
        search for each vector to be inserted, and inserting it with edges to
        all nodes traversed during the search. Reverse edges are also inserted
        and robustPrune is applied to improve graph quality. The index_params
        struct controls the degree of the final graph.
    
        The following distance metrics are supported:
            - L2Expanded
    
        Parameters
        ----------
        index_params : IndexParams object
        dataset : CUDA array interface compliant matrix shape (n_samples, dim)
            Supported dtype [float32, int8, uint8]
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Returns
        -------
        index: cuvs.vamana.Index
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> from cuvs.neighbors import vamana
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> build_params = vamana.IndexParams(
        ...     metric="sqeuclidean", graph_degree=64, visited_size=128)
        >>> index = vamana.build(build_params, dataset)
        >>> # Serialize index to file for later use with CPU DiskANN
        >>> vamana.save("my_index.bin", index)
        
    """
def save(*args, resources = None, **kwargs):
    """
    save(filename, Index index, bool include_dataset=True, resources=None)
    
        Saves the index to a file.
    
        Matches the file format used by the DiskANN open-source repository,
        allowing cross-compatibility.
    
        Parameters
        ----------
        filename : string
            Name of the file.
        index : Index
            Trained Vamana index.
        include_dataset : bool
            Whether or not to write out the dataset along with the index. Including
            the dataset in the serialized index will use extra disk space, and
            might not be desired if you already have a copy of the dataset on
            disk.
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.neighbors import vamana
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> # Build index
        >>> index = vamana.build(vamana.IndexParams(graph_degree=64,
        ...                                         visited_size=128), dataset)
        >>> # Serialize and save the vamana index
        >>> vamana.save("my_index.bin", index)
        
    """
DISTANCE_NAMES: dict = {1: 'euclidean', 0: 'sqeuclidean', 3: 'cityblock', 6: 'inner_product', 7: 'chebyshev', 8: 'canberra', 2: 'cosine', 9: 'minkowski', 10: 'correlation', 11: 'jaccard', 12: 'hellinger', 14: 'braycurtis', 15: 'jensenshannon', 16: 'hamming', 17: 'kl_divergence', 18: 'russellrao', 19: 'dice', 20: 'bitwise_hamming'}
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {'build (line 169)': '\n    Build the Vamana index from the dataset for efficient search.\n\n    The build utilities the Vamana insertion-based algorithm to create\n    the graph. The algorithm starts with an empty graph and iteratively\n    inserts batches of nodes. Each batch involves performing a greedy\n    search for each vector to be inserted, and inserting it with edges to\n    all nodes traversed during the search. Reverse edges are also inserted\n    and robustPrune is applied to improve graph quality. The index_params\n    struct controls the degree of the final graph.\n\n    The following distance metrics are supported:\n        - L2Expanded\n\n    Parameters\n    ----------\n    index_params : IndexParams object\n    dataset : CUDA array interface compliant matrix shape (n_samples, dim)\n        Supported dtype [float32, int8, uint8]\n    {resources_docstring}\n\n    Returns\n    -------\n    index: cuvs.vamana.Index\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import vamana\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> build_params = vamana.IndexParams(\n    ...     metric="sqeuclidean", graph_degree=64, visited_size=128)\n    >>> index = vamana.build(build_params, dataset)\n    >>> # Serialize index to file for later use with CPU DiskANN\n    >>> vamana.save("my_index.bin", index)\n    ', 'save (line 239)': '\n    Saves the index to a file.\n\n    Matches the file format used by the DiskANN open-source repository,\n    allowing cross-compatibility.\n\n    Parameters\n    ----------\n    filename : string\n        Name of the file.\n    index : Index\n        Trained Vamana index.\n    include_dataset : bool\n        Whether or not to write out the dataset along with the index. Including\n        the dataset in the serialized index will use extra disk space, and\n        might not be desired if you already have a copy of the dataset on\n        disk.\n    {resources_docstring}\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.neighbors import vamana\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> # Build index\n    >>> index = vamana.build(vamana.IndexParams(graph_degree=64,\n    ...                                         visited_size=128), dataset)\n    >>> # Serialize and save the vamana index\n    >>> vamana.save("my_index.bin", index)\n    '}
