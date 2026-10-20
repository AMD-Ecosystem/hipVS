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
  :keywords: Nearest-Neighbors, Information-Retrieval, Similarity-Search, GPU, Distance, RAPIDS, AMD Data Science

.. _hipvs:

********************************************************************
hipVS documentation
********************************************************************

hipVS is a GPU-accelerated vector search library for AMD GPUs, enabling high-performance approximate and exact nearest-neighbor (ANN) search and clustering workloads. It is part of the AMD Data Science toolkit, an open-source software collection for high-performance data science applications. Forked from the RAPIDS® cuVS project and aligned with RAPIDS 26.06, hipVS brings state-of-the-art vector similarity capabilities to the :doc:`ROCm <rocm:index>`/:doc:`HIP <hip:index>` software stack and AMD platforms. It is designed to minimize porting friction for users familiar with cuVS by maintaining consistent directory structure, file organization, and API naming. For more information, see :doc:`What is hipVS? <./what-is-hipVS>`

Key Features in hipVS v2.0.0:

* Approximate and exact nearest-neighbor search implementations, including:

  - Brute-force (exact k-NN)
  - IVF-Flat (Inverted File with flat storage)
  - IVF-PQ (Inverted File with Product Quantization)
  - CAGRA (GPU-optimized graph-based ANN)
  - HNSW (Hierarchical Navigable Small World)
  - NN-Descent (approximate k-NN graph construction)
  - Vamana/DiskANN (graph-based index construction)

* C++, C, Python, and Rust interfaces for integration in both production systems and rapid-prototyping environments.
* API compatibility with RAPIDS cuVS, enabling drop-in portability for existing workflows on AMD hardware.
* Optimized GPU performance for fast index building, low-latency queries, and high-throughput search.
* Interoperability with other AMD Data Science components such as hipRAFT, hipDF, and hipGRAPH for end-to-end data-science and AI pipelines.

New in v2.0.0 (aligned with RAPIDS 26.06):

* **New algorithms and capabilities**

  - PCA preprocessor for dimensionality reduction before index building and search
  - HNSW ACE (Augmented Core Extraction) build method with automatic partition count derivation
  - CAGRA merge with pre-filtering support for filtered merging of multiple CAGRA indices
  - L1 (Manhattan) distance support for the CAGRA iterative-search graph build path

* **IVF-PQ enhancements**

  - Build factories with precomputed centroids and codebooks (C++, C, and Python APIs)
  - New ``transform`` function to encode vectors into PQ codes without building or searching an index
  - Flat (uninterleaved) PQ code export for interoperability and debugging
  - Standalone PQ preprocessing API

* **Extended data type support**

  - Float16 (half) support for CAGRA ``extend`` (adding vectors to existing indices)
  - NN-Descent fp16 support exposed through the Python API
  - Improved fp32 distance dispatch by dimension in NN-Descent

* **KMeans improvements**

  - Batch processing for training on datasets larger than GPU memory
  - Inertia computation for balanced KMeans with weighted inertia option

* **Spectral clustering and embedding**

  - ``all_neighbors`` graph construction for spectral embedding
  - Dataset API for spectral clustering
  - Optional seed parameter for reproducible spectral embeddings
  - Configurable non-zero element type (``nnz_t``) for sparse matrices

* **DiskANN / Vamana**

  - Configurable DRAM usage in the DiskANN SSD wrapper
  - Rust API bindings for Vamana index build and search

* **Language binding expansions**

  - Rust ``Index::search`` changed to borrow (``&self``) for repeated searches without re-creation
  - PQ list data accessible from Python for advanced inspection

* **Performance improvements**

  - Reduced CAGRA binary size via runtime parameterization of search kernels
  - Improved IVF-PQ coarse-batch workspace estimation for lower memory overhead
  - Better multi-GPU replicated-mode search batching and load balancing
  - Kernel deduplication for reduced compile times and binary size

* **Build and platform updates**

  - Upgraded to C++20
  - Updated FAISS dependency to 1.13.0
  - HNSW GPU hierarchy is now the default build behavior

The hipVS code is open and hosted at `https://github.com/AMD-Ecosystem/hipVS <https://github.com/AMD-Ecosystem/hipVS>`_.

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


To contribute to the documentation refer to `Contributing to AMD Data Science  <https://rocm.docs.amd.com/projects/datascience/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/projects/datascience/en/latest/about/license.html>`_ page.
