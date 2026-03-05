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

Multi-GPU Nearest Neighbors
===========================

The Multi-GPU (SNMG - single-node multi-GPUs) C API provides a set of functions to deploy ANN indexes across multiple GPUs for improved performance and scalability.

.. role:: py(code)
   :language: c
   :class: highlight

Common Types and Enums
======================

Common types and enums used across multi-GPU ANN algorithms.

``#include <cuvs/neighbors/mg_common.h>``

.. doxygengroup:: mg_c_common_types
    :project: cuvs
    :members:
    :content-only:

Multi-GPU IVF-Flat
==================

The Multi-GPU IVF-Flat method extends the IVF-Flat ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

``#include <cuvs/neighbors/mg_ivf_flat.h>``

IVF-Flat Index Build Parameters
-------------------------------

.. doxygengroup:: mg_ivf_flat_c_index_params
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Search Parameters
--------------------------------

.. doxygengroup:: mg_ivf_flat_c_search_params
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index
--------------

.. doxygengroup:: mg_ivf_flat_c_index
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Build
--------------------

.. doxygengroup:: mg_ivf_flat_c_index_build
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Search
---------------------

.. doxygengroup:: mg_ivf_flat_c_index_search
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Extend
---------------------

.. doxygengroup:: mg_ivf_flat_c_index_extend
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Serialize
------------------------

.. doxygengroup:: mg_ivf_flat_c_index_serialize
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Deserialize
---------------------------

.. doxygengroup:: mg_ivf_flat_c_index_deserialize
    :project: cuvs
    :members:
    :content-only:

IVF-Flat Index Distribute
--------------------------

.. doxygengroup:: mg_ivf_flat_c_index_distribute
    :project: cuvs
    :members:
    :content-only:

Multi-GPU IVF-PQ
=================

The Multi-GPU IVF-PQ method extends the IVF-PQ ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

``#include <cuvs/neighbors/mg_ivf_pq.h>``

IVF-PQ Index Build Parameters
-----------------------------

.. doxygengroup:: mg_ivf_pq_c_index_params
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Search Parameters
------------------------------

.. doxygengroup:: mg_ivf_pq_c_search_params
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index
------------

.. doxygengroup:: mg_ivf_pq_c_index
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Build
------------------

.. doxygengroup:: mg_ivf_pq_c_index_build
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Search
-------------------

.. doxygengroup:: mg_ivf_pq_c_index_search
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Extend
-------------------

.. doxygengroup:: mg_ivf_pq_c_index_extend
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Serialize
----------------------

.. doxygengroup:: mg_ivf_pq_c_index_serialize
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Deserialize
------------------------

.. doxygengroup:: mg_ivf_pq_c_index_deserialize
    :project: cuvs
    :members:
    :content-only:

IVF-PQ Index Distribute
-----------------------

.. doxygengroup:: mg_ivf_pq_c_index_distribute
    :project: cuvs
    :members:
    :content-only:

Multi-GPU CAGRA
================

The Multi-GPU CAGRA method extends the CAGRA graph-based ANN algorithm to work across multiple GPUs. It provides two distribution modes: replicated (for higher throughput) and sharded (for handling larger datasets).

``#include <cuvs/neighbors/mg_cagra.h>``

CAGRA Index Build Parameters
----------------------------

.. doxygengroup:: mg_cagra_c_index_params
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Search Parameters
-----------------------------

.. doxygengroup:: mg_cagra_c_search_params
    :project: cuvs
    :members:
    :content-only:

CAGRA Index
-----------

.. doxygengroup:: mg_cagra_c_index
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Build
-----------------

.. doxygengroup:: mg_cagra_c_index_build
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Search
------------------

.. doxygengroup:: mg_cagra_c_index_search
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Extend
------------------

.. doxygengroup:: mg_cagra_c_index_extend
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Serialize
---------------------

.. doxygengroup:: mg_cagra_c_index_serialize
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Deserialize
-----------------------

.. doxygengroup:: mg_cagra_c_index_deserialize
    :project: cuvs
    :members:
    :content-only:

CAGRA Index Distribute
----------------------

.. doxygengroup:: mg_cagra_c_index_distribute
    :project: cuvs
    :members:
    :content-only:
