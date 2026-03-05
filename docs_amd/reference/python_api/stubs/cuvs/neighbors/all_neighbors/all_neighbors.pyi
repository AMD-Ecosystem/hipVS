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
from cuvs.common.mg_resources import MultiGpuResources
from cuvs.common.resources import Resources
from cuvs.neighbors.common import _check_input_array
from cuvs.neighbors.ivf_pq.ivf_pq import IndexParams as IvfPqIndexParams
from cuvs.neighbors.nn_descent.nn_descent import IndexParams as NNDescentIndexParams
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['AllNeighborsParams', 'DISTANCE_TYPES', 'IvfPqIndexParams', 'MultiGpuResources', 'NNDescentIndexParams', 'Resources', 'auto_convert_output', 'build', 'check_cuvs', 'cuda_interruptible', 'device_ndarray', 'np', 'wrap_array']
class AllNeighborsParams:
    """
    AllNeighborsParams(algo=u'nn_descent', *, overlap_factor=2, n_clusters=1, metric=u'sqeuclidean', ivf_pq_params=None, nn_descent_params=None)
    
        Parameters for all-neighbors k-NN graph building.
    
        Parameters
        ----------
        algo : str or cuvsAllNeighborsAlgo
            Algorithm to use for local k-NN graph building.
            Options: "brute_force", "ivf_pq", "nn_descent"
        overlap_factor : int, default=2
            Number of clusters each point is assigned to (must be < n_clusters)
        n_clusters : int, default=1
            Number of clusters/batches to partition the dataset into
            (> overlap_factor). Use n_clusters>1 to distribute the work
            across GPUs.
        metric : str or cuvsDistanceType, default="sqeuclidean"
            Distance metric to use for graph construction
        ivf_pq_params : cuvs.neighbors.ivf_pq.IndexParams, optional
            IVF-PQ specific parameters (used when algo="ivf_pq")
        nn_descent_params : cuvs.neighbors.nn_descent.IndexParams, optional
            NN-Descent specific parameters (used when algo="nn_descent")
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        AllNeighborsParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        AllNeighborsParams.__setstate_cython__(self, __pyx_state)
        """
    def get_handle(self):
        """
        AllNeighborsParams.get_handle(self)
        Get a pointer to the underlying C object.
        """
def __reduce_cython__(self):
    """
    AllNeighborsParams.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    AllNeighborsParams.__setstate_cython__(self, __pyx_state)
    """
def build(dataset, k, params, *, indices = None, distances = None, core_distances = None, alpha = 1.0, resources = None):
    """
        All-neighbors allows building an approximate all-neighbors knn graph.
        Given a full dataset, it finds nearest neighbors for all the training
        vectors in the dataset.
    
        Parameters
        ----------
        dataset : array_like
            Training dataset to build the k-NN graph for. Can be provided
            on host (for multi-GPU build) or device (for single-GPU build).
            Host vs device location is automatically detected.
            Supported dtype: float32
        k : int
            Number of nearest neighbors to find for each point
        params : AllNeighborsParams
            Parameters object containing all build settings including algorithm
            choice and algorithm-specific parameters.
        indices : array_like, optional
            Optional output buffer for indices [num_rows x k] on device
            (int64). If not provided, will be allocated automatically.
        distances : array_like, optional
            Optional output buffer for distances [num_rows x k] on device
            (float32)
        core_distances : array_like, optional
            Optional output buffer for core distances [num_rows] on device
            (float32). Requires distances parameter to be provided.
        alpha : float, default=1.0
            Mutual-reachability scaling; used only when core_distances is
            provided
        resources : Resources or MultiGpuResources, optional
            CUDA resources to use for the operation. If not provided, a default
            Resources object will be created. Use MultiGpuResources to enable
            multi-GPU execution across multiple devices.
    
        Returns
        -------
        indices : array_like
            k-NN indices for each point [num_rows x k], always on device.
            If indices buffer was provided, returns the same array filled
            with results.
        distances : array_like or None
            k-NN distances if distances buffer was provided, None otherwise
        core_distances : array_like or None
            Core distances if core_distances buffer was provided, None otherwise
    """
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
__test__: dict = {}
