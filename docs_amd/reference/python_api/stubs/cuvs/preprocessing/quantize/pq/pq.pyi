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
from cuvs.common.device_tensor_view import DeviceTensorView
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['DeviceTensorView', 'PQ_KMEANS_NAMES', 'PQ_KMEANS_TYPES', 'Quantizer', 'QuantizerParams', 'auto_convert_output', 'auto_sync_resources', 'build', 'check_cuvs', 'device_ndarray', 'inverse_transform', 'np', 'transform', 'wrap_array']
class Quantizer:
    """
    
        Defines and stores Product Quantizer upon training
    
        The quantization is performed by a linear mapping of an interval in the
        float data type to the full range of the quantized int type.
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Quantizer.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Quantizer.__setstate_cython__(self, __pyx_state)
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class QuantizerParams:
    """
    QuantizerParams(pq_bits=8, *, pq_dim=0, use_subspaces=True, use_vq=False, vq_n_centers=0, kmeans_n_iters=25, pq_kmeans_type='kmeans_balanced', max_train_points_per_pq_code=256, max_train_points_per_vq_cluster=1024)
    
    Parameters for product quantization
    
    Parameters
    ----------
    pq_bits: int
        specifies the bit length of the vector element after compression by PQ
        possible values: within [4, 16]
    pq_dim: int
        specifies the dimensionality of the vector after compression by PQ
    use_subspaces: bool
        specifies whether to use subspaces for product quantization (PQ).
        When true, one PQ codebook is used for each subspace. Otherwise, a
        single PQ codebook is used.
    use_vq: bool
        specifies whether to use Vector Quantization (KMeans) before product
        quantization (PQ).
    vq_n_centers: int
        specifies the number of centers for the vector quantizer.
        When zero, an optimal value is selected using a heuristic.
        When one, only product quantization is used.
    kmeans_n_iters: int
        specifies the number of iterations searching for kmeans centers
    pq_kmeans_type: str
        specifies the type of kmeans algorithm to use for PQ training
        possible values: "kmeans", "kmeans_balanced"
    max_train_points_per_pq_code: int
        specifies the max number of data points to use per PQ code during PQ
        codebook training. Using more data points per PQ code may increase the
        quality of PQ codebook but may also increase the build time.
    max_train_points_per_vq_cluster: int
        specifies the max number of data points to use per VQ cluster.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        QuantizerParams.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        QuantizerParams.__setstate_cython__(self, __pyx_state)
        """
def __reduce_cython__(self):
    """
    Quantizer.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    Quantizer.__setstate_cython__(self, __pyx_state)
    """
def build(*args, resources = None, **kwargs):
    """
    build(QuantizerParams params, dataset, resources=None)
    
    Builds a Product Quantizer to be used later for quantizing
    the dataset.
    
    Parameters
    ----------
    params : QuantizerParams object
    dataset : row major dataset on host or device memory. FP32
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    quantizer: cuvs.preprocessing.quantize.pq.Quantizer
    
    Examples
    --------
    
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import pq
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = pq.build(params, dataset)
    >>> transformed, _ = pq.transform(quantizer, dataset)
    """
def inverse_transform(*args, resources = None, **kwargs):
    """
    inverse_transform(Quantizer quantizer, codes, output=None, vq_labels=None, resources=None)
    
    Applies Product Quantization inverse transform to given codes
    
    Parameters
    ----------
    quantizer : trained Quantizer object
    codes : row major device codes to inverse transform. uint8
    output : optional preallocated output memory, on device memory
    vq_labels : optional VQ labels when VQ is used, on device memory
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    output : Original dataset reconstructed from quantized codes
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import pq
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16, use_vq=True)
    >>> quantizer = pq.build(params, dataset)
    >>> transformed, vq_labels = pq.transform(quantizer, dataset)
    >>> reconstructed = pq.inverse_transform(quantizer, transformed, vq_labels=vq_labels)
    """
def transform(*args, resources = None, **kwargs):
    """
    transform(Quantizer quantizer, dataset, codes_output=None, vq_labels=None, resources=None)
    
    Applies Product Quantization transform to given dataset
    
    Parameters
    ----------
    quantizer : trained Quantizer object
    dataset : row major dataset on host or device memory. FP32
    codes_output : optional preallocated output memory, on device memory
    vq_labels : optional preallocated output memory for VQ labels, on device memory
    resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.
    
    Returns
    -------
    codes_output : transformed dataset quantized into a uint8
    vq_labels : VQ labels when VQ is used, None otherwise
    
    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import pq
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = pq.build(params, dataset)
    >>> transformed, _ = pq.transform(quantizer, dataset)
    """
PQ_KMEANS_NAMES: dict = {0: 'kmeans', 1: 'kmeans_balanced'}
PQ_KMEANS_TYPES: dict = {'kmeans': 0, 'kmeans_balanced': 1}
__test__: dict = {'build (line 197)': '\n    Builds a Product Quantizer to be used later for quantizing\n    the dataset.\n\n    Parameters\n    ----------\n    params : QuantizerParams object\n    dataset : row major dataset on host or device memory. FP32\n    {resources_docstring}\n\n    Returns\n    -------\n    quantizer: cuvs.preprocessing.quantize.pq.Quantizer\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing.quantize import pq\n    >>> n_samples = 5000\n    >>> n_features = 64\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)\n    >>> quantizer = pq.build(params, dataset)\n    >>> transformed, _ = pq.transform(quantizer, dataset)\n    ', 'transform (line 245)': '\n    Applies Product Quantization transform to given dataset\n\n    Parameters\n    ----------\n    quantizer : trained Quantizer object\n    dataset : row major dataset on host or device memory. FP32\n    codes_output : optional preallocated output memory, on device memory\n    vq_labels : optional preallocated output memory for VQ labels, on device memory\n    {resources_docstring}\n\n    Returns\n    -------\n    codes_output : transformed dataset quantized into a uint8\n    vq_labels : VQ labels when VQ is used, None otherwise\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing.quantize import pq\n    >>> n_samples = 5000\n    >>> n_features = 64\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)\n    >>> quantizer = pq.build(params, dataset)\n    >>> transformed, _ = pq.transform(quantizer, dataset)\n    ', 'inverse_transform (line 312)': '\n    Applies Product Quantization inverse transform to given codes\n\n    Parameters\n    ----------\n    quantizer : trained Quantizer object\n    codes : row major device codes to inverse transform. uint8\n    output : optional preallocated output memory, on device memory\n    vq_labels : optional VQ labels when VQ is used, on device memory\n    {resources_docstring}\n\n    Returns\n    -------\n    output : Original dataset reconstructed from quantized codes\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing.quantize import pq\n    >>> n_samples = 5000\n    >>> n_features = 64\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16, use_vq=True)\n    >>> quantizer = pq.build(params, dataset)\n    >>> transformed, vq_labels = pq.transform(quantizer, dataset)\n    >>> reconstructed = pq.inverse_transform(quantizer, transformed, vq_labels=vq_labels)\n    '}
