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

Multi-GPU IVF-PQ
================

Multi-GPU IVF-PQ extends the IVF-PQ (Inverted File with Product Quantization) algorithm to work across multiple GPUs, providing improved scalability and performance for large-scale vector search. It supports both replicated and sharded distribution modes.

.. role:: py(code)
   :language: python
   :class: highlight

.. note::
   **IMPORTANT**: Multi-GPU IVF-PQ requires all data (datasets, queries, output arrays) to be in host memory (CPU).
   If using CuPy/device arrays, transfer to host with ``array.get()`` or ``cp.asnumpy(array)`` before use.

Index build parameters
######################

.. autoapiclass:: cuvs.neighbors.mg.ivf_pq.IndexParams
    :members:

Index search parameters
#######################

.. autoapiclass:: cuvs.neighbors.mg.ivf_pq.SearchParams
    :members:

Index
#####

.. autoapiclass:: cuvs.neighbors.mg.ivf_pq.Index
    :members:

Index build
###########

.. autoapifunction:: cuvs.neighbors.mg.ivf_pq.build

Index search
############

.. autoapifunction:: cuvs.neighbors.mg.ivf_pq.search

Index extend
############

.. autoapifunction:: cuvs.neighbors.mg.ivf_pq.extend

Index save
##########

.. autoapifunction:: cuvs.neighbors.mg.ivf_pq.save

Index load
##########

.. autoapifunction:: cuvs.neighbors.mg.ivf_pq.load

Index distribute
################

.. autoapifunction:: cuvs.neighbors.mg.ivf_pq.distribute
