from __future__ import annotations
import builtins as __builtins__
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['Quantizer', 'QuantizerParams', 'auto_convert_output', 'auto_sync_resources', 'check_cuvs', 'device_ndarray', 'inverse_transform', 'np', 'train', 'transform', 'wrap_array']
class Quantizer:
    """

        Defines and stores scalar for quantisation upon training

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
    def __repr__(*args, **kwargs):
        ...
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Quantizer.__setstate_cython__(self, __pyx_state)
        """
class QuantizerParams:
    """
    QuantizerParams(quantile=None, *)

        Parameters for scalar quantization

        Parameters
        ----------
        quantile: float
            specifies how many outliers at top & bottom will be ignored
            needs to be within range of (0, 1]

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
def inverse_transform(*args, resources = None, **kwargs):
    """
    inverse_transform(Quantizer quantizer, dataset, output=None, resources=None)

        Perform inverse quantization step on previously quantized dataset

        Note that depending on the chosen data types train dataset the conversion
        is not lossless.

        Parameters
        ----------
        quantizer : trained Quantizer object
        dataset : row major host or device dataset to transform
        output : optional preallocated output memory, on host or device
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Returns
        -------
        output : transformed dataset with scalar quantization reversed

    """
def train(*args, resources = None, **kwargs):
    """
    train(QuantizerParams params, dataset, resources=None)

        Initializes a scalar quantizer to be used later for quantizing the dataset.

        Parameters
        ----------
        params : QuantizerParams object
        dataset : row major host or device dataset
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Returns
        -------
        quantizer: cuvs.preprocessing.quantize.scalar.Quantizer

        Examples
        --------

        >>> import cupy as cp
        >>> from cuvs.preprocessing.quantize import scalar
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> params = scalar.QuantizerParams(quantile=0.99)
        >>> quantizer = scalar.train(params, dataset)
        >>> transformed = scalar.transform(quantizer, dataset)

    """
def transform(*args, resources = None, **kwargs):
    """
    transform(Quantizer quantizer, dataset, output=None, resources=None)

        Applies quantization transform to given dataset

        Parameters
        ----------
        quantizer : trained Quantizer object
        dataset : row major host or device dataset to transform
        output : optional preallocated output memory, on host or device memory
        resources : Optional cuVS Resource handle for reusing CUDA resources.
            If Resources aren't supplied, CUDA resources will be
            allocated inside this function and synchronized before the
            function exits. If resources are supplied, you will need to
            explicitly synchronize yourself by calling `resources.sync()`
            before accessing the output.

        Returns
        -------
        output : transformed dataset quantized into a int8

        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.preprocessing.quantize import scalar
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.random_sample((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> params = scalar.QuantizerParams(quantile=0.99)
        >>> quantizer = scalar.train(params, dataset)
        >>> transformed = scalar.transform(quantizer, dataset)

    """
__test__: dict = {'train (line 85)': '\n    Initializes a scalar quantizer to be used later for quantizing the dataset.\n\n    Parameters\n    ----------\n    params : QuantizerParams object\n    dataset : row major host or device dataset\n    {resources_docstring}\n\n    Returns\n    -------\n    quantizer: cuvs.preprocessing.quantize.scalar.Quantizer\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing.quantize import scalar\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> params = scalar.QuantizerParams(quantile=0.99)\n    >>> quantizer = scalar.train(params, dataset)\n    >>> transformed = scalar.transform(quantizer, dataset)\n    ', 'transform (line 134)': '\n    Applies quantization transform to given dataset\n\n    Parameters\n    ----------\n    quantizer : trained Quantizer object\n    dataset : row major host or device dataset to transform\n    output : optional preallocated output memory, on host or device memory\n    {resources_docstring}\n\n    Returns\n    -------\n    output : transformed dataset quantized into a int8\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing.quantize import scalar\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.random_sample((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> params = scalar.QuantizerParams(quantile=0.99)\n    >>> quantizer = scalar.train(params, dataset)\n    >>> transformed = scalar.transform(quantizer, dataset)\n    '}
