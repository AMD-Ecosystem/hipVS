from __future__ import annotations
import builtins as __builtins__
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.device_ndarray import device_ndarray
from pylibraft.common.outputs import auto_convert_output
__all__: list[str] = ['auto_convert_output', 'auto_sync_resources', 'check_cuvs', 'device_ndarray', 'np', 'transform', 'wrap_array']
def transform(*args, resources = None, **kwargs):
    """
    transform(dataset, output=None, resources=None)

        Applies binary quantization transform to given dataset

        This applies binary quantization to a dataset, changing any positive
        values to a bitwise 1. This is useful for searching with the
        BitwiseHamming distance type.

        Parameters
        ----------
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
        output : transformed dataset quantized into a uint8

        Examples
        --------
        >>> import cupy as cp
        >>> from cuvs.preprocessing.quantize import binary
        >>> from cuvs.neighbors import cagra
        >>> n_samples = 50000
        >>> n_features = 50
        >>> dataset = cp.random.standard_normal((n_samples, n_features),
        ...                                   dtype=cp.float32)
        >>> transformed = binary.transform(dataset)
        >>>
        >>> # build a cagra index on the binarized data
        >>> params = cagra.IndexParams(metric="bitwise_hamming",
        ...                            build_algo="iterative_cagra_search")
        >>> idx = cagra.build(params, transformed)

    """
__test__: dict = {'transform (line 30)': '\n    Applies binary quantization transform to given dataset\n\n    This applies binary quantization to a dataset, changing any positive\n    values to a bitwise 1. This is useful for searching with the\n    BitwiseHamming distance type.\n\n    Parameters\n    ----------\n    dataset : row major host or device dataset to transform\n    output : optional preallocated output memory, on host or device memory\n    {resources_docstring}\n\n    Returns\n    -------\n    output : transformed dataset quantized into a uint8\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cuvs.preprocessing.quantize import binary\n    >>> from cuvs.neighbors import cagra\n    >>> n_samples = 50000\n    >>> n_features = 50\n    >>> dataset = cp.random.standard_normal((n_samples, n_features),\n    ...                                   dtype=cp.float32)\n    >>> transformed = binary.transform(dataset)\n    >>>\n    >>> # build a cagra index on the binarized data\n    >>> params = cagra.IndexParams(metric="bitwise_hamming",\n    ...                            build_algo="iterative_cagra_search")\n    >>> idx = cagra.build(params, transformed)\n    '}
