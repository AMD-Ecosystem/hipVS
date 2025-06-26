#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
#
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# cython: language_level=3

import functools

# TODO(HIP-PYTHON) Once https://github.com/AMD-AI/hip-python/pull/3 has merged
# and made public, uncomment the line below and remove the line underneath
# from cuda.bindings.cyruntime cimport cudaStream_t

from cuda.ccuda cimport cudaStream_t

from cuvs.common.c_api cimport (
    cuvsResources_t,
    cuvsResourcesCreate,
    cuvsResourcesDestroy,
    cuvsStreamSet,
    cuvsStreamSync,
)

from cuvs.common.exceptions import check_cuvs


cdef class Resources:
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

    def __cinit__(self, stream=None):
        check_cuvs(cuvsResourcesCreate(&self.c_obj))
        if stream:
            check_cuvs(cuvsStreamSet(self.c_obj, <cudaStream_t>stream))

    def sync(self):
        check_cuvs(cuvsStreamSync(self.c_obj))

    def get_c_obj(self):
        """
        Return the pointer to the underlying c_obj as a size_t
        """
        return <size_t> self.c_obj

    def __dealloc__(self):
        check_cuvs(cuvsResourcesDestroy(self.c_obj))


_resources_param_string = """
     resources : Optional cuVS Resource handle for reusing CUDA resources.
        If Resources aren't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If resources are supplied, you will need to
        explicitly synchronize yourself by calling `resources.sync()`
        before accessing the output.
""".strip()


def auto_sync_resources(f):
    """Decorator to automatically call sync on a cuVS Resources object when
    it isn't passed to a function.

    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default resources for the function, and
    call sync on that resources when the function exits.

    This will also insert the appropriate docstring for the resources parameter
    """

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None
        resources = resources if resources is not None else Resources()

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_resources_param_string
    )
    return wrapper
