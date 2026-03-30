..
    MIT License

    Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

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

.. meta::
  :description: hipVS documentation and API reference library
  :keywords: Nearest-Neighbors, Information-Retrieval, Similarity-Search, GPU, Distance, RAPIDS, ROCm-DS

.. _hipvs:

********************************************************************
hipVS documentation
********************************************************************

hipVS is a GPU-accelerated vector search library for AMD GPUs, enabling high-performance approximate and exact nearest-neighbor (ANN) search and clustering workloads. It is part of the AMD ROCm™ Data Science toolkit (ROCm-DS), an open-source software collection for high-performance data science applications. Forked from the NVIDIA® RAPIDS® cuVS project and aligned with RAPIDS 25.10, hipVS brings state-of-the-art vector similarity capabilities to the :doc:`ROCm <rocm:index>`/:doc:`HIP <hip:index>` software stack and AMD platforms. It is designed to minimize porting friction for users familiar with cuVS by maintaining consistent directory structure, file organization, and API naming. For more information, see :doc:`What is hipVS? <./what-is-hipVS>`

Key Features in hipVS v1.0.0:

* Approximate and exact nearest-neighbor search implementations, including:

  - HNSW (Hierarchical Navigable Small World)
  - IVF-PQ (Inverted File with Product Quantization)
  - Brute-force (exact k-NN)

* C++, C, Python, and Rust interfaces for integration in both production systems and rapid-prototyping environments.
* API compatibility with RAPIDS cuVS, enabling drop-in portability for existing workflows on AMD hardware.
* Optimized GPU performance for fast index building, low-latency queries, and high-throughput search.
* Interoperability with other ROCm-DS components such as hipRAFT, hipDF, and hipGRAPH for end-to-end data-science and AI pipelines.

The hipVS code is open and hosted at `https://github.com/ROCm-DS/hipVS <https://github.com/ROCm-DS/hipVS>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`System requirements <install/system-requirements>`
    * :doc:`Installing hipVS <install/install>`
    * :doc:`Building hipVS <install/build>`

  .. grid-item-card:: How to

    * :doc:`Use hipVS <how-to/using-hipVS>`

  .. grid-item-card:: API reference

    * :ref:`C++ API reference <hipvs-cpp>`
    * :ref:`Python API reference <hipvs-python>`
    * :ref:`C API reference <hipvs-c>`
    * :ref:`Rust API reference <hipvs-rust>`


To contribute to the documentation refer to `Contributing to ROCm-DS  <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/about/license.html>`_ page.
