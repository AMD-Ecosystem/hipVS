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

HNSW
====

This is a wrapper for hnswlib, to load a CAGRA index as an immutable HNSW index. The loaded HNSW index is only compatible in hipVS, and can be searched using wrapper functions.

.. role:: py(code)
   :language: python
   :class: highlight

Index search parameters
#######################

.. autoapiclass:: cuvs.neighbors.hnsw.SearchParams
    :members:

Index
#####

.. autoapiclass:: cuvs.neighbors.hnsw.Index
    :members:

Index Conversion
################

.. autoapifunction:: cuvs.neighbors.hnsw.from_cagra

Index search
############

.. autoapifunction:: cuvs.neighbors.hnsw.search

Index save
##########

.. autoapifunction:: cuvs.neighbors.hnsw.save

Index load
##########

.. autoapifunction:: cuvs.neighbors.hnsw.load

Index extend
############

.. autoapifunction:: cuvs.neighbors.hnsw.extend
