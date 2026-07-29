.. meta::
  :description: hipVS documentation and API reference library
  :keywords: Nearest-Neighbors, Information-Retrieval, Similarity-Search, GPU, Distance, RAPIDS, AMD Data Science

.. _hipvs:

********************************************************************
hipVS documentation
********************************************************************

hipVS is a GPU-accelerated vector search library for AMD GPUs, enabling high-performance approximate and exact nearest-neighbor (ANN) search and clustering workloads. It is part of the AMD Data Science toolkit, an open-source software collection for high-performance data science applications. Forked from the NVIDIA® RAPIDS® cuVS project and aligned with RAPIDS 25.02, hipVS brings state-of-the-art vector similarity capabilities to the :doc:`HIP <hip:index>`/:doc:`ROCm <rocm:index>` software stack and AMD platforms. It is designed to minimize porting friction for users familiar with cuVS by maintaining consistent directory structure, file organization, and API naming. For more information, see :doc:`What is hipVS? <./what-is-hipVS>`

Key Features in hipVS v0.1.0:

* Approximate and exact nearest-neighbor search implementations, including:

  - HNSW (Hierarchical Navigable Small World)
  - IVF-PQ (Inverted File with Product Quantization)
  - Brute-force (exact k-NN)

* C++, C, Python, and Rust interfaces for integration in both production systems and rapid-prototyping environments.
* API compatibility with RAPIDS cuVS, enabling drop-in portability for existing workflows on AMD hardware.
* Optimized GPU performance for fast index building, low-latency queries, and high-throughput search.
* Interoperability with other Data Science components such as hipRAFT, hipDF, and hipGRAPH for end-to-end data-science and AI pipelines.

The hipVS code is open and hosted at `https://github.com/AMD-Ecosystem/hipVS <https://github.com/AMD-Ecosystem/hipVS>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`Installing hipVS <install/install>`
    * :doc:`Building hipVS <install/build>`

  .. grid-item-card:: How to

    * :doc:`Use hipVS <how-to/using-hipVS>`

  .. grid-item-card:: API reference

    * :ref:`C++ API reference <hipvs-cpp>`
    * :ref:`Python API reference <hipvs-python>`
    * :ref:`C API reference <hipvs-c>`
    * :ref:`Rust API reference <hipvs-rust>`


To contribute to the documentation refer to `Contributing to AMD Data Science  <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/about/license.html>`_ page.
