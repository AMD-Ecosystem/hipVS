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
from _frozen_importlib import FitTransformOutput
import builtins as __builtins__
from collections import namedtuple
import cupy as cp
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['FitOutput', 'FitTransformOutput', 'Params', 'SOLVER_NAMES', 'SOLVER_TYPES', 'auto_convert_output', 'auto_sync_resources', 'check_cuvs', 'cp', 'device_ndarray', 'fit', 'fit_transform', 'inverse_transform', 'namedtuple', 'np', 'transform', 'wrap_array']
class Params:
    """
    Params(n_components=None, *, copy=None, whiten=None, algorithm=None, tol=None, n_iterations=None)
    
    Parameters for PCA decomposition.
    
    Parameters
    ----------
    n_components : int
        Number of principal components to keep (default: 1).
    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X) then transform(X) will not yield the expected results;
        use fit_transform(X) instead (default: True).
    whiten : bool
        When True the component vectors are multiplied by the square root
        of n_samples and divided by the singular values to ensure
        uncorrelated outputs with unit component-wise variances
        (default: False).
    algorithm : str
        Solver algorithm. One of ``"cov_eig_dq"`` (divide-and-conquer)
        or ``"cov_eig_jacobi"`` (Jacobi) (default: ``"cov_eig_dq"``).
    tol : float
        Tolerance for singular values, used by the Jacobi solver
        (default: 0.0).
    n_iterations : int
        Number of iterations for the Jacobi solver (default: 15).
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Params.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Params.__setstate_cython__(self, __pyx_state)
        """
def __reduce_cython__(self):
    """
    Params.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    Params.__setstate_cython__(self, __pyx_state)
    """
def _to_f_order(ary):
    """
    _to_f_order(ary)
    
    Ensure a device array is Fortran-contiguous (col-major).
    """
def fit(*args, resources = None, **kwargs):
    """
    fit(Params params, X, resources=None)
    
    Compute PCA (fit only).
    
    Computes the principal components, explained variances, singular
    values, and column means from the input data.
    
    Parameters
    ----------
    params : Params
        PCA parameters. ``params.copy`` should be True if you intend
        to reuse *X* after this call.
    X : device array-like, shape (n_samples, n_features), float32
        Input data (will be converted to col-major device memory).
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    FitOutput
        Named tuple with fields: ``components``, ``explained_var``,
        ``explained_var_ratio``, ``singular_vals``, ``mu``,
        ``noise_vars``.
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8, copy=True)
    >>> result = pca.fit(params, X)
    >>> result.components.shape
    (8, 32)
    """
def fit_transform(*args, resources = None, **kwargs):
    """
    fit_transform(Params params, X, resources=None)
    
    Compute PCA and transform the input data in a single operation.
    
    Parameters
    ----------
    params : Params
        PCA parameters.
    X : device array-like, shape (n_samples, n_features), float32
        Input data (will be converted to col-major device memory).
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    FitTransformOutput
        Named tuple with fields: ``trans_input``, ``components``,
        ``explained_var``, ``explained_var_ratio``, ``singular_vals``,
        ``mu``, ``noise_vars``.
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8)
    >>> result = pca.fit_transform(params, X)
    >>> result.trans_input.shape
    (500, 8)
    """
def inverse_transform(*args, resources = None, **kwargs):
    """
    inverse_transform(Params params, trans_input, components, singular_vals, mu, output=None, resources=None)
    
    Transform data from the PCA eigenspace back to the original space.
    
    Parameters
    ----------
    params : Params
        PCA parameters (must match those used during fit).
    trans_input : device array-like, shape (n_samples, n_components)
        Transformed data from :func:`transform` or :func:`fit_transform`.
    components : device array-like, shape (n_components, n_features)
        Principal components from a prior fit.
    singular_vals : device array-like, shape (n_components,)
        Singular values from a prior fit.
    mu : device array-like, shape (n_features,)
        Column means from a prior fit.
    output : optional device array, shape (n_samples, n_features)
        Pre-allocated output buffer (col-major, float32).
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    output : device array, shape (n_samples, n_features)
        Reconstructed data.
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8)
    >>> result = pca.fit_transform(params, X)
    >>> reconstructed = pca.inverse_transform(
    ...     params, result.trans_input, result.components,
    ...     result.singular_vals, result.mu)
    """
def transform(*args, resources = None, **kwargs):
    """
    transform(Params params, X, components, singular_vals, mu, trans_input=None, resources=None)
    
    Transform data into the PCA eigenspace.
    
    Uses previously computed principal components from :func:`fit` or
    :func:`fit_transform`.
    
    Parameters
    ----------
    params : Params
        PCA parameters (must match those used during fit).
    X : device array-like, shape (n_samples, n_features), float32
        Data to transform.
    components : device array-like, shape (n_components, n_features)
        Principal components from a prior fit.
    singular_vals : device array-like, shape (n_components,)
        Singular values from a prior fit.
    mu : device array-like, shape (n_features,)
        Column means from a prior fit.
    trans_input : optional device array, shape (n_samples, n_components)
        Pre-allocated output buffer (col-major, float32).
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    trans_input : device array, shape (n_samples, n_components)
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8, copy=True)
    >>> result = pca.fit(params, X)
    >>> transformed = pca.transform(params, X, result.components,
    ...                             result.singular_vals, result.mu)
    """
SOLVER_NAMES: dict = {0: 'cov_eig_dq', 1: 'cov_eig_jacobi'}
SOLVER_TYPES: dict = {'cov_eig_dq': 0, 'cov_eig_jacobi': 1}
__test__: dict = {'fit (line 126)': '\n    Compute PCA (fit only).\n\n    Computes the principal components, explained variances, singular\n    values, and column means from the input data.\n\n    Parameters\n    ----------\n    params : Params\n        PCA parameters. ``params.copy`` should be True if you intend\n        to reuse *X* after this call.\n    X : device array-like, shape (n_samples, n_features), float32\n        Input data (will be converted to col-major device memory).\n    {resources_docstring}\n\n    Returns\n    -------\n    FitOutput\n        Named tuple with fields: ``components``, ``explained_var``,\n        ``explained_var_ratio``, ``singular_vals``, ``mu``,\n        ``noise_vars``.\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing import pca\n    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)\n    >>> params = pca.Params(n_components=8, copy=True)\n    >>> result = pca.fit(params, X)\n    >>> result.components.shape\n    (8, 32)\n    ', 'fit_transform (line 199)': '\n    Compute PCA and transform the input data in a single operation.\n\n    Parameters\n    ----------\n    params : Params\n        PCA parameters.\n    X : device array-like, shape (n_samples, n_features), float32\n        Input data (will be converted to col-major device memory).\n    {resources_docstring}\n\n    Returns\n    -------\n    FitTransformOutput\n        Named tuple with fields: ``trans_input``, ``components``,\n        ``explained_var``, ``explained_var_ratio``, ``singular_vals``,\n        ``mu``, ``noise_vars``.\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing import pca\n    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)\n    >>> params = pca.Params(n_components=8)\n    >>> result = pca.fit_transform(params, X)\n    >>> result.trans_input.shape\n    (500, 8)\n    ', 'transform (line 273)': '\n    Transform data into the PCA eigenspace.\n\n    Uses previously computed principal components from :func:`fit` or\n    :func:`fit_transform`.\n\n    Parameters\n    ----------\n    params : Params\n        PCA parameters (must match those used during fit).\n    X : device array-like, shape (n_samples, n_features), float32\n        Data to transform.\n    components : device array-like, shape (n_components, n_features)\n        Principal components from a prior fit.\n    singular_vals : device array-like, shape (n_components,)\n        Singular values from a prior fit.\n    mu : device array-like, shape (n_features,)\n        Column means from a prior fit.\n    trans_input : optional device array, shape (n_samples, n_components)\n        Pre-allocated output buffer (col-major, float32).\n    {resources_docstring}\n\n    Returns\n    -------\n    trans_input : device array, shape (n_samples, n_components)\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing import pca\n    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)\n    >>> params = pca.Params(n_components=8, copy=True)\n    >>> result = pca.fit(params, X)\n    >>> transformed = pca.transform(params, X, result.components,\n    ...                             result.singular_vals, result.mu)\n    ', 'inverse_transform (line 351)': '\n    Transform data from the PCA eigenspace back to the original space.\n\n    Parameters\n    ----------\n    params : Params\n        PCA parameters (must match those used during fit).\n    trans_input : device array-like, shape (n_samples, n_components)\n        Transformed data from :func:`transform` or :func:`fit_transform`.\n    components : device array-like, shape (n_components, n_features)\n        Principal components from a prior fit.\n    singular_vals : device array-like, shape (n_components,)\n        Singular values from a prior fit.\n    mu : device array-like, shape (n_features,)\n        Column means from a prior fit.\n    output : optional device array, shape (n_samples, n_features)\n        Pre-allocated output buffer (col-major, float32).\n    {resources_docstring}\n\n    Returns\n    -------\n    output : device array, shape (n_samples, n_features)\n        Reconstructed data.\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing import pca\n    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)\n    >>> params = pca.Params(n_components=8)\n    >>> result = pca.fit_transform(params, X)\n    >>> reconstructed = pca.inverse_transform(\n    ...     params, result.trans_input, result.components,\n    ...     result.singular_vals, result.mu)\n    '}
