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
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['DISTANCE_NAMES', 'DISTANCE_TYPES', 'SUPPORTED_DISTANCES', 'auto_convert_output', 'auto_sync_resources', 'check_cuvs', 'device_ndarray', 'np', 'pairwise_distance', 'wrap_array']
def pairwise_distance(*args, resources = None, **kwargs):
    """
    pairwise_distance(X, Y, out=None, metric=u'euclidean', p=2.0, resources=None)
    
        Compute pairwise distances between X and Y
    
        Valid values for metric:
            ["euclidean", "l2", "l1", "cityblock", "inner_product",
             "chebyshev", "canberra", "lp", "hellinger", "jensenshannon",
             "kl_divergence", "russellrao", "minkowski", "correlation",
             "cosine"]
    
        Parameters
        ----------
    
        X : CUDA array interface compliant matrix shape (m, k)
        Y : CUDA array interface compliant matrix shape (n, k)
        out : Optional writable CUDA array interface matrix shape (m, n)
        metric : string denoting the metric type (default="euclidean")
        p : metric parameter (currently used only for "minkowski")
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> from cuvs.distance import pairwise_distance
        >>> n_samples = 5000
        >>> n_features = 50
        >>> in1 = cp.random.random_sample((n_samples, n_features),
        ...                               dtype=cp.float32)
        >>> in2 = cp.random.random_sample((n_samples, n_features),
        ...                               dtype=cp.float32)
        >>> output = pairwise_distance(in1, in2, metric="euclidean")
        
    """
DISTANCE_NAMES: dict = {1: 'euclidean', 0: 'sqeuclidean', 3: 'cityblock', 6: 'inner_product', 7: 'chebyshev', 8: 'canberra', 2: 'cosine', 9: 'minkowski', 10: 'correlation', 11: 'jaccard', 12: 'hellinger', 14: 'braycurtis', 15: 'jensenshannon', 16: 'hamming', 17: 'kl_divergence', 18: 'russellrao', 19: 'dice', 20: 'bitwise_hamming'}
DISTANCE_TYPES: dict = {'l2': 1, 'sqeuclidean': 0, 'euclidean': 1, 'l1': 3, 'cityblock': 3, 'inner_product': 6, 'chebyshev': 7, 'canberra': 8, 'cosine': 2, 'lp': 9, 'correlation': 10, 'jaccard': 11, 'hellinger': 12, 'braycurtis': 14, 'jensenshannon': 15, 'hamming': 16, 'kl_divergence': 17, 'minkowski': 9, 'russellrao': 18, 'dice': 19, 'bitwise_hamming': 20}
SUPPORTED_DISTANCES: list = ['euclidean', 'l1', 'cityblock', 'l2', 'inner_product', 'chebyshev', 'minkowski', 'canberra', 'kl_divergence', 'correlation', 'russellrao', 'hellinger', 'lp', 'hamming', 'jensenshannon', 'cosine', 'sqeuclidean']
__test__: dict = {'pairwise_distance (line 60)': '\n    Compute pairwise distances between X and Y\n\n    Valid values for metric:\n        ["euclidean", "l2", "l1", "cityblock", "inner_product",\n         "chebyshev", "canberra", "lp", "hellinger", "jensenshannon",\n         "kl_divergence", "russellrao", "minkowski", "correlation",\n         "cosine"]\n\n    Parameters\n    ----------\n\n    X : CUDA array interface compliant matrix shape (m, k)\n    Y : CUDA array interface compliant matrix shape (n, k)\n    out : Optional writable CUDA array interface matrix shape (m, n)\n    metric : string denoting the metric type (default="euclidean")\n    p : metric parameter (currently used only for "minkowski")\n    {resources_docstring}\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.distance import pairwise_distance\n    >>> n_samples = 5000\n    >>> n_features = 50\n    >>> in1 = cp.random.random_sample((n_samples, n_features),\n    ...                               dtype=cp.float32)\n    >>> in2 = cp.random.random_sample((n_samples, n_features),\n    ...                               dtype=cp.float32)\n    >>> output = pairwise_distance(in1, in2, metric="euclidean")\n    '}
