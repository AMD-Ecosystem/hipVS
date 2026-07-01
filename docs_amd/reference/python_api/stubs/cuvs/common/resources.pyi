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
import functools as functools
__all__: list[str] = ['Resources', 'auto_sync_resources', 'check_cuvs', 'functools']
class Resources:
    """
    
        Resources  is a lightweight python wrapper around the corresponding
        C++ class of resources exposed by RAFT's C++ interface. Refer to
        the header file raft/core/resources.hpp for interface level
        details of this struct.
    
        Parameters
        ----------
        stream : Optional stream to use for ordering CUDA instructions
    
        Examples
        --------
    
        Basic usage:
    
        >>> from cuvs.common import Resources
        >>> handle = Resources()
        >>>
        >>> # call algos here
        >>>
        >>> # final sync of all work launched in the stream of this handle
        >>> handle.sync()
    
        Using a cuPy stream with cuVS Resources:
    
        >>> import cupy
        >>> from cuvs.common import Resources
        >>>
        >>> cupy_stream = cupy.cuda.Stream()
        >>> handle = Resources(stream=cupy_stream.ptr)
        
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        """
        Resources.__reduce_cython__(self)
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        Resources.__setstate_cython__(self, __pyx_state)
        """
    def get_c_obj(self):
        """
        Resources.get_c_obj(self)
        
        Return the pointer to the underlying c_obj as a size_t
        """
    def sync(self):
        """
        Resources.sync(self)
        """
def __reduce_cython__(self):
    """
    Resources.__reduce_cython__(self)
    """
def __setstate_cython__(self, __pyx_state):
    """
    Resources.__setstate_cython__(self, __pyx_state)
    """
def auto_sync_resources(f):
    """
    auto_sync_resources(f)
    
    Decorator to automatically call sync on a cuVS Resources object when
    it isn't passed to a function.
    
    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default resources for the function, and
    call sync on that resources when the function exits.
    
    This will also insert the appropriate docstring for the resources parameter
    """
__test__: dict = {}
_resources_param_string: str = "resources : Optional cuVS Resource handle for reusing CUDA resources.\n        If Resources aren't supplied, CUDA resources will be\n        allocated inside this function and synchronized before the\n        function exits. If resources are supplied, you will need to\n        explicitly synchronize yourself by calling `resources.sync()`\n        before accessing the output."
