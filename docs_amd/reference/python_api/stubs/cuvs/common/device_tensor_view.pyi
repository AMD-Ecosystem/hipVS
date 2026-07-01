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
from cuvs.common.cydlpack import dl_data_type_to_numpy
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
__all__: list[str] = ['DeviceTensorView', 'auto_sync_resources', 'check_cuvs', 'dl_data_type_to_numpy', 'np', 'wrap_array']
class DeviceTensorView:
    """
    DeviceTensorView(array_like=None)
    
    A __cuda_array_interface__ compatible tensor on device memory
    
    This aims to be a lightweight wrapper around device_matrix_view and
    device_vector_view objects in the C++ layer of cuvs - and lets us
    access memory internals of cuvs without copying into temporary
    storage.
    
    Since this support the `__cuda_array_interface__` protocol, this object
    gives us easy interopability with other libraries such as cupy, pytorch,
    and cudf - without adding any dependencies on these libraries inside cuvs.
    
    Since cuvs also understands CAI, DeviceTensorView objects can also be
    passed directly to cuvs python functions.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        DeviceTensorView.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        DeviceTensorView.__setstate_cython__(self, __pyx_state)
        """
    @staticmethod
    def copy_to_host(*args, resources = None, **kwargs):
        """
        DeviceTensorView.copy_to_host(self, resources=None)
        
        copies a device matrix to a numpy array
        
        Parameters
        ----------
        resources : Optional cuVS Resource handle for reusing CUDA resources.
                If Resources aren't supplied, CUDA resources will be
                allocated inside this function and synchronized before the
                function exits. If resources are supplied, you will need to
                explicitly synchronize yourself by calling `resources.sync()`
                before accessing the output.
        
        Returns
        -------
        tensor: np.array
            This tensor copied into host memory
        """
    @staticmethod
    def slice_rows(*args, resources = None, **kwargs):
        """
        DeviceTensorView.slice_rows(self, start, end, resources=None)
        
        Slices rows from this tensor
        
        Slices a subset, and returns in a new DeviceTensorView without
        copying data.
        
        Parameters
        ----------
        start: int
            the index of the first row to slice
        end: int
            the index of the last row to slice
        resources : Optional cuVS Resource handle for reusing CUDA resources.
                If Resources aren't supplied, CUDA resources will be
                allocated inside this function and synchronized before the
                function exits. If resources are supplied, you will need to
                explicitly synchronize yourself by calling `resources.sync()`
                before accessing the output.
        
        Returns
        -------
        tensor: DeviceTensorView
            A non-owning view of the rows sliced from this tensor
        """
    def get_handle(self):
        """
        DeviceTensorView.get_handle(self)
        """
def __reduce_cython__(self):
    """
    DeviceTensorView.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    DeviceTensorView.__setstate_cython__(self, __pyx_state)
    """
__test__: dict = {}
