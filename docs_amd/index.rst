.. meta::
  :description: hipVS documentation and API reference library
  :keywords: Nearest-Neighbors, Information-Retrieval, Similarity-Search, GPU, Distance, RAPIDS, ROCm-DS

.. _hipvs:

********************************************************************
hipVS documentation
********************************************************************

hipVS is a port of the RAPIDS `cuvs <https://github.com/rapidsai/cuvs>`_ library. It
follows the directory structure, file naming and API naming as closely as possible to
minimize porting friction for developers using both projects. It contains state-of-the-art implementations of several algorithms for running approximate nearest neighbors and clustering on the GPU. The primary goal of hipVS is to simplify the use of GPUs for vector similarity search and clustering. For more information, see :doc:`What is hipVS <./what-is-hipVS>`.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`Installing hipVS <install/install>`
    * :doc:`Building hipVS <install/build>`

  .. grid-item-card:: Examples

    * :doc:`Examples <examples_readme>`

  .. grid-item-card:: API reference

    * :ref:`C++ API reference <hipvs-cpp>`
    * :ref:`Python API reference <hipvs-python>`
    * :ref:`C API reference <hipvs-c>`


To contribute to the documentation refer to `Contributing to ROCm-DS  <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/about/license.html>`_ page.
