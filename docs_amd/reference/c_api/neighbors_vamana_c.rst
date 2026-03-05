..
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Vamana
======

Vamana is the graph construction algorithm behind the well-known DiskANN vector search solution. The cuVS implementation of Vamana/DiskANN is a custom GPU-accelerated version of the algorithm that aims to reduce index construction time using NVIDIA GPUs. hipVS is the AMD fork of cuVS, bringing the same GPU-accelerated Vamana/DiskANN index construction to AMD GPUs via HIP/ROCm.


.. role:: py(code)
   :language: c
   :class: highlight

``#include <cuvs/neighbors/vamana.h>``

Index build parameters
----------------------

.. doxygengroup:: vamana_c_index_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: vamana_c_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: vamana_c_index_build
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: vamana_c_index_serialize
    :project: cuvs
    :members:
    :content-only:
