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
from cuvs.neighbors.common import _check_input_array
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
__all__: list[str] = ['Prefilter', 'from_bitmap', 'from_bitset', 'no_filter', 'np', 'wrap_array']
class Prefilter:
    """
    Prefilter(cuvsFilter prefilter, parent=None)
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Prefilter.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Prefilter.__setstate_cython__(self, __pyx_state)
        """
def __reduce_cython__(self):
    """
    Prefilter.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    Prefilter.__setstate_cython__(self, __pyx_state)
    """
def from_bitmap(bitmap):
    """
    from_bitmap(bitmap)
    
        Create a pre-filter from an array with type of uint32.
    
        Parameters
        ----------
        bitmap : numpy.ndarray
            An array with type of `uint32` where each bit in the array corresponds
            to if a sample and query pair is greenlit (not filtered) or filtered.
            The array is row-major, meaning the bits are ordered by rows first.
            Each bit in a `uint32` element represents a different sample-query
            pair.
    
            - Bit value of 1: The sample-query pair is greenlit (allowed).
            - Bit value of 0: The sample-query pair is filtered.
    
        Returns
        -------
        filter : cuvs.neighbors.filters.Prefilter
            An instance of `Prefilter` that can be used to filter neighbors
            based on the given bitmap.
        {resources_docstring}
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> import numpy as np
        >>> from cuvs.neighbors import filters
        >>>
        >>> n_samples = 50000
        >>> n_queries = 1000
        >>>
        >>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)
        >>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)
        >>> prefilter = filters.from_bitmap(bitmap)
        
    """
def from_bitset(bitset):
    """
    from_bitset(bitset)
    
        Create a pre-filter from an array with type of uint32.
    
        Parameters
        ----------
        bitset : numpy.ndarray
            An array with type of `uint32` where each bit in the array
            corresponds to if a sample is greenlit (not filtered) or filtered.
            Each bit in a `uint32` element represents a different sample of
            the dataset.
    
            - Bit value of 1: The sample is greenlit (allowed).
            - Bit value of 0: The sample pair is filtered.
    
        Returns
        -------
        filter : cuvs.neighbors.filters.Prefilter
            An instance of `Prefilter` that can be used to filter neighbors
            based on the given bitset.
        {resources_docstring}
    
        Examples
        --------
    
        >>> import cupy as cp
        >>> import numpy as np
        >>> from cuvs.neighbors import filters
        >>>
        >>> n_samples = 50000
        >>> n_queries = 1000
        >>>
        >>> n_bitset = np.ceil(n_samples / 32).astype(int)
        >>> bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)
        >>> prefilter = filters.from_bitset(bitset)
        
    """
def no_filter():
    """
    no_filter()
    
        Create a default pre-filter which filters nothing.
        
    """
__test__: dict = {'from_bitmap (line 50)': '\n    Create a pre-filter from an array with type of uint32.\n\n    Parameters\n    ----------\n    bitmap : numpy.ndarray\n        An array with type of `uint32` where each bit in the array corresponds\n        to if a sample and query pair is greenlit (not filtered) or filtered.\n        The array is row-major, meaning the bits are ordered by rows first.\n        Each bit in a `uint32` element represents a different sample-query\n        pair.\n\n        - Bit value of 1: The sample-query pair is greenlit (allowed).\n        - Bit value of 0: The sample-query pair is filtered.\n\n    Returns\n    -------\n    filter : cuvs.neighbors.filters.Prefilter\n        An instance of `Prefilter` that can be used to filter neighbors\n        based on the given bitmap.\n    {resources_docstring}\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> import numpy as np\n    >>> from cuvs.neighbors import filters\n    >>>\n    >>> n_samples = 50000\n    >>> n_queries = 1000\n    >>>\n    >>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)\n    >>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)\n    >>> prefilter = filters.from_bitmap(bitmap)\n    ', 'from_bitset (line 100)': '\n    Create a pre-filter from an array with type of uint32.\n\n    Parameters\n    ----------\n    bitset : numpy.ndarray\n        An array with type of `uint32` where each bit in the array\n        corresponds to if a sample is greenlit (not filtered) or filtered.\n        Each bit in a `uint32` element represents a different sample of\n        the dataset.\n\n        - Bit value of 1: The sample is greenlit (allowed).\n        - Bit value of 0: The sample pair is filtered.\n\n    Returns\n    -------\n    filter : cuvs.neighbors.filters.Prefilter\n        An instance of `Prefilter` that can be used to filter neighbors\n        based on the given bitset.\n    {resources_docstring}\n\n    Examples\n    --------\n\n    >>> import cupy as cp\n    >>> import numpy as np\n    >>> from cuvs.neighbors import filters\n    >>>\n    >>> n_samples = 50000\n    >>> n_queries = 1000\n    >>>\n    >>> n_bitset = np.ceil(n_samples / 32).astype(int)\n    >>> bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)\n    >>> prefilter = filters.from_bitset(bitset)\n    '}
