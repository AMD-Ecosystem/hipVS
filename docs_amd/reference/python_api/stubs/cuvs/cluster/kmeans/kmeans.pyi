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
from _frozen_importlib import FitOutput
from _frozen_importlib import PredictOutput
import builtins as __builtins__
from collections import namedtuple
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
__all__: list[str] = ['DISTANCE_NAMES', 'DISTANCE_TYPES', 'FitOutput', 'INIT_METHOD_NAMES', 'INIT_METHOD_TYPES', 'KMeansParams', 'PredictOutput', 'auto_convert_output', 'auto_sync_resources', 'cai_wrapper', 'check_cuvs', 'cluster_cost', 'cuda_interruptible', 'device_ndarray', 'fit', 'namedtuple', 'np', 'predict', 'warnings', 'wrap_array']
class KMeansParams:
    """
    KMeansParams(metric=None, *, n_clusters=None, init_method=None, max_iter=None, tol=None, n_init=None, oversampling_factor=None, batch_samples=None, batch_centroids=None, inertia_check=None, init_size=None, streaming_batch_size=None, hierarchical=None, hierarchical_n_iters=None)
    
    Hyper-parameters for the kmeans algorithm
    
    Parameters
    ----------
    metric : str
        String denoting the metric type.
    n_clusters : int
        The number of clusters to form as well as the number of centroids
        to generate
    init_method : str
        Method for initializing clusters. One of:
        "KMeansPlusPlus" : Use scalable k-means++ algorithm to select initial
        cluster centers
        "Random" : Choose 'n_clusters' observations at random from the input
        data
        "Array" : Use centroids as initial cluster centers
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run
    tol : float
        Relative tolerance with regards to inertia to declare convergence.
    n_init : int
        Number of instance k-means algorithm will be run with different seeds
    oversampling_factor : double
        Oversampling factor for use in the k-means|| algorithm
    batch_samples : int
        Number of samples to process in each batch for tiled 1NN computation.
        Useful to optimize/control memory footprint. Default tile is
        [batch_samples x n_clusters].
    batch_centroids : int
        Number of centroids to process in each batch. If 0, uses n_clusters.
    inertia_check : bool
        Deprecated and ignored. Will be
        removed in a future release. Inertia-based convergence checking
        always runs.
    init_size : int
        Number of samples to draw for KMeansPlusPlus initialization with
        host (out-of-core) data. When set to 0, uses the heuristic
        min(3 * n_clusters, n_samples). Default: 0.
    streaming_batch_size : int
        Number of samples to process per GPU batch when fitting with host
        (numpy) data. When set to 0, defaults to n_samples (process all
        at once). Only used by the batched (host-data) code path. Reducing
        streaming_batch_size can help reduce GPU memory pressure but increases
        overhead as the number of times centroid adjustments are computed
        increases.
    
        Default: 0 (process all data at once).
    hierarchical : bool
        Whether to use hierarchical (balanced) kmeans or not
    hierarchical_n_iters : int
        For hierarchical k-means , defines the number of training iterations
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        KMeansParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        KMeansParams.__setstate_cython__(self, __pyx_state)
        """
def __reduce_cython__(self):
    """
    KMeansParams.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    KMeansParams.__setstate_cython__(self, __pyx_state)
    """
def cluster_cost(*args, resources = None, **kwargs):
    """
    cluster_cost(X, centroids, resources=None)
    
    Compute cluster cost given an input matrix and existing centroids
    
    Parameters
    ----------
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Input CUDA array interface compliant matrix shape
                    (n_clusters, k)
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    inertia : float
        The cluster cost between the input matrix and existing centroids
    
    Examples
    --------
    
    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import cluster_cost
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)
    
    >>> centroids = cp.random.random_sample((n_clusters, n_features),
    ...                                      dtype=cp.float32)
    
    >>> inertia = cluster_cost(X, centroids)
    """
def fit(*args, resources = None, **kwargs):
    """
    fit(KMeansParams params, X, centroids=None, sample_weights=None, resources=None)
    
    Find clusters with the k-means algorithm
    
    When X is a device array (CUDA array interface), standard on-device
    k-means is used.  When X is a host array (numpy ndarray or
    ``__array_interface__``), data is streamed to the GPU in batches
    controlled by ``params.streaming_batch_size``. For large host datasets, consider
    reducing ``streaming_batch_size`` to reduce GPU memory usage.
    
    Parameters
    ----------
    
    params : KMeansParams
        Parameters to use to fit KMeans model.  For host data,
        ``params.streaming_batch_size`` controls how many samples are sent to the
        GPU per batch.
    X : array-like
        Training instances, shape (m, k).  Accepts both device arrays
        (cupy / CUDA array interface) and host arrays (numpy).
    centroids : Optional writable CUDA array interface compliant matrix
                shape (n_clusters, k)
    sample_weights : Optional weights per observation.  Must reside on
                     the same memory space as X (device or host).
                     default: None
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    centroids : raft.device_ndarray
        The computed centroids for each cluster
    inertia : float
       Sum of squared distances of samples to their closest cluster center
    n_iter : int
        The number of iterations used to fit the model
    
    Examples
    --------
    
    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import fit, KMeansParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)
    
    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)
    
    Host-data (batched) example:
    
    >>> import numpy as np
    >>> X_host = np.random.random((10_000_000, 128)).astype(np.float32)
    >>> params = KMeansParams(n_clusters=1000, streaming_batch_size=1_000_000)
    >>> centroids, inertia, n_iter = fit(params, X_host)
    """
def predict(*args, resources = None, **kwargs):
    """
    predict(KMeansParams params, X, centroids, sample_weights=None, labels=None, normalize_weight=True, resources=None)
    
    Predict clusters with the k-means algorithm
    
    Parameters
    ----------
    
    params : KMeansParams
        Parameters to used in fitting KMeans model
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : CUDA array interface compliant matrix, calculated by fit
                shape (n_clusters, k)
    sample_weights : Optional input CUDA array interface compliant matrix shape
                     (n_clusters, 1) default: None
    labels : Optional preallocated CUDA array interface matrix shape (m, 1)
        to hold the output
    normalize_weight: bool
        True if the weights should be normalized
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    labels : raft.device_ndarray
        The label for each datapoint in X
    inertia : float
       Sum of squared distances of samples to their closest cluster center
    
    Examples
    --------
    
    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import fit, predict, KMeansParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)
    
    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)
    >>>
    >>> labels, inertia = predict(params, X, centroids)
    """
DISTANCE_NAMES: dict = {1: 'euclidean', 0: 'sqeuclidean', 3: 'cityblock', 6: 'inner_product', 7: 'chebyshev', 8: 'canberra', 2: 'cosine', 9: 'minkowski', 10: 'correlation', 11: 'jaccard', 12: 'hellinger', 14: 'braycurtis', 15: 'jensenshannon', 16: 'hamming', 17: 'kl_divergence', 18: 'russellrao', 19: 'dice', 20: 'bitwise_hamming'}
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
INIT_METHOD_NAMES: dict = {0: 'KMeansPlusPlus', 1: 'Random', 2: 'Array'}
INIT_METHOD_TYPES: dict = {'KMeansPlusPlus': 0, 'Random': 1, 'Array': 2}
__test__: dict = {'fit (line 219)': '\n    Find clusters with the k-means algorithm\n\n    When X is a device array (CUDA array interface), standard on-device\n    k-means is used.  When X is a host array (numpy ndarray or\n    ``__array_interface__``), data is streamed to the GPU in batches\n    controlled by ``params.streaming_batch_size``. For large host datasets, consider\n    reducing ``streaming_batch_size`` to reduce GPU memory usage.\n\n    Parameters\n    ----------\n\n    params : KMeansParams\n        Parameters to use to fit KMeans model.  For host data,\n        ``params.streaming_batch_size`` controls how many samples are sent to the\n        GPU per batch.\n    X : array-like\n        Training instances, shape (m, k).  Accepts both device arrays\n        (cupy / CUDA array interface) and host arrays (numpy).\n    centroids : Optional writable CUDA array interface compliant matrix\n                shape (n_clusters, k)\n    sample_weights : Optional weights per observation.  Must reside on\n                     the same memory space as X (device or host).\n                     default: None\n    {resources_docstring}\n\n    Returns\n    -------\n    centroids : raft.device_ndarray\n        The computed centroids for each cluster\n    inertia : float\n       Sum of squared distances of samples to their closest cluster center\n    n_iter : int\n        The number of iterations used to fit the model\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>>\n    >>> from cuvs.cluster.kmeans import fit, KMeansParams\n    >>>\n    >>> n_samples = 5000\n    >>> n_features = 50\n    >>> n_clusters = 3\n    >>>\n    >>> X = cp.random.random_sample((n_samples, n_features),\n    ...                             dtype=cp.float32)\n\n    >>> params = KMeansParams(n_clusters=n_clusters)\n    >>> centroids, inertia, n_iter = fit(params, X)\n\n    Host-data (batched) example:\n\n    >>> import numpy as np\n    >>> X_host = np.random.random((10_000_000, 128)).astype(np.float32)\n    >>> params = KMeansParams(n_clusters=1000, streaming_batch_size=1_000_000)\n    >>> centroids, inertia, n_iter = fit(params, X_host)\n    ', 'predict (line 356)': '\n    Predict clusters with the k-means algorithm\n\n    Parameters\n    ----------\n\n    params : KMeansParams\n        Parameters to used in fitting KMeans model\n    X : Input CUDA array interface compliant matrix shape (m, k)\n    centroids : CUDA array interface compliant matrix, calculated by fit\n                shape (n_clusters, k)\n    sample_weights : Optional input CUDA array interface compliant matrix shape\n                     (n_clusters, 1) default: None\n    labels : Optional preallocated CUDA array interface matrix shape (m, 1)\n        to hold the output\n    normalize_weight: bool\n        True if the weights should be normalized\n    {resources_docstring}\n\n    Returns\n    -------\n    labels : raft.device_ndarray\n        The label for each datapoint in X\n    inertia : float\n       Sum of squared distances of samples to their closest cluster center\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>>\n    >>> from cuvs.cluster.kmeans import fit, predict, KMeansParams\n    >>>\n    >>> n_samples = 5000\n    >>> n_features = 50\n    >>> n_clusters = 3\n    >>>\n    >>> X = cp.random.random_sample((n_samples, n_features),\n    ...                             dtype=cp.float32)\n\n    >>> params = KMeansParams(n_clusters=n_clusters)\n    >>> centroids, inertia, n_iter = fit(params, X)\n    >>>\n    >>> labels, inertia = predict(params, X, centroids)\n    ', 'cluster_cost (line 447)': '\n    Compute cluster cost given an input matrix and existing centroids\n\n    Parameters\n    ----------\n    X : Input CUDA array interface compliant matrix shape (m, k)\n    centroids : Input CUDA array interface compliant matrix shape\n                    (n_clusters, k)\n    {resources_docstring}\n\n    Returns\n    -------\n    inertia : float\n        The cluster cost between the input matrix and existing centroids\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>>\n    >>> from cuvs.cluster.kmeans import cluster_cost\n    >>>\n    >>> n_samples = 5000\n    >>> n_features = 50\n    >>> n_clusters = 3\n    >>>\n    >>> X = cp.random.random_sample((n_samples, n_features),\n    ...                             dtype=cp.float32)\n\n    >>> centroids = cp.random.random_sample((n_clusters, n_features),\n    ...                                      dtype=cp.float32)\n\n    >>> inertia = cluster_cost(X, centroids)\n    '}
