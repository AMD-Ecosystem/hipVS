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
import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
__all__: list[str] = ['MultiGpuResources', 'auto_sync_multi_gpu_resources', 'check_cuvs', 'np', 'wrap_array']
class MultiGpuResources:
    """
    
        Multi-GPU Resources is a lightweight python wrapper around the
        corresponding C++ class of multi-GPU resources exposed by RAFT's C++
        interface. This class provides a handle for multi-GPU operations across
        all available GPUs.
    
        Parameters
        ----------
        stream : int, optional
            A CUDA stream pointer to use for this resource handle. If None, a
            default stream will be used.
        device_ids : list of int, optional
            A list of device IDs to use for multi-GPU operations. If None, all
            available GPUs will be used.
    
        Examples
        --------
    
        Basic usage:
    
        >>> from cuvs.common import MultiGpuResources
        >>> handle = MultiGpuResources()
        >>>
        >>> # call multi-GPU algos here
        >>>
        >>> # final sync of all work launched in the stream of this handle
        >>> handle.sync()
    
        Using a cuPy stream with cuVS Multi-GPU Resources:
    
        >>> import cupy
        >>> from cuvs.common import MultiGpuResources
        >>>
        >>> cupy_stream = cupy.cuda.Stream()
        >>> handle = MultiGpuResources(stream=cupy_stream.ptr)
    
        Using specific device IDs:
    
        >>> from cuvs.common import MultiGpuResources
        >>> handle = MultiGpuResources(device_ids=[0])
        >>>
        >>> # call multi-GPU algos here
        >>>
        >>> handle.sync()
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        MultiGpuResources.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        MultiGpuResources.__setstate_cython__(self, __pyx_state)
        """
    def get_c_obj(self):
        """
        MultiGpuResources.get_c_obj(self)
        
        Return the pointer to the underlying c_obj as a size_t
        """
    def set_memory_pool(self, percent_of_free_memory):
        """
        MultiGpuResources.set_memory_pool(self, percent_of_free_memory)
        
        Set a memory pool on all devices managed by these resources.
        
        Parameters
        ----------
        percent_of_free_memory : int
            Percentage of free device memory to allocate for the pool.
        
        Examples
        --------
        >>> from cuvs.common import MultiGpuResources
        >>> handle = MultiGpuResources()
        >>> handle.set_memory_pool(80)  # Use 80% of free memory
        """
    def sync(self):
        """
        MultiGpuResources.sync(self)
        """
def __reduce_cython__(self):
    """
    MultiGpuResources.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    MultiGpuResources.__setstate_cython__(self, __pyx_state)
    """
def auto_sync_multi_gpu_resources(f):
    """
    auto_sync_multi_gpu_resources(f)
    
    Decorator to automatically call sync on a cuVS Multi-GPU Resources
    object when it isn't passed to a function.
    
    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default multi-GPU resources for the function,
    and call sync on that resources when the function exits.
    
    This will also insert the appropriate docstring for the resources
    parameter
    """
__test__: dict = {'MultiGpuResources.set_memory_pool (line 93)': '\n        Set a memory pool on all devices managed by these resources.\n\n        Parameters\n        ----------\n        percent_of_free_memory : int\n            Percentage of free device memory to allocate for the pool.\n\n        Examples\n        --------\n        >>> from cuvs.common import MultiGpuResources\n        >>> handle = MultiGpuResources()\n        >>> handle.set_memory_pool(80)  # Use 80% of free memory\n        '}
_multi_gpu_resources_param_string: str = "resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA resources.\n        If Multi-GPU Resources aren't supplied, CUDA resources will be\n        allocated inside this function and synchronized before the\n        function exits. If resources are supplied, you will need to\n        explicitly synchronize yourself by calling `resources.sync()`\n        before accessing the output."
