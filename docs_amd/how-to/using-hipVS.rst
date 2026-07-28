..
    MIT License

    Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
   :keywords: Machine-Learning, Vector Search, Primitives,GPU, RAPIDS, AMD Data Science

Using hipVS examples
=====================

Examples demonstrating the use of the hipVS library are provided in the ``examples`` folder.
Currently, examples targeting C, C++, and Python bindings are available.  These
example projects can be used as templates to build your own application using
hipVS, or to add hipVS to existing projects.

Building the examples
----------------------

To build the examples, use the provided ``build.sh`` script. This is a bash script
that calls the appropriate CMake commands, so you can look into it to see the typical
CMake based build workflow.

The dependencies of the examples project, including hipVS, are automatically fetched and built
by the CMake scripts, but there are some basic pre-requisites and dependencies that need to be
satisfied for a successful build. Please refer to the
:doc:`Build and installation <../install/build>`
documentation for instructions on setting up your developer environment for building hipVS.

The individual example project directories can be copied as a starting point to build a new
standalone application using hipVS. Make sure to also copy the ``<hipvs_source>/examples/cmake``
folder, as it contains the functionality to fetch and build hipVS.

.. code-block:: bash

   # project directory for a new hipVS based application
   HIPVS_SRC_DIR=...
   PROJECT_SRC_DIR=...

   # copy the cpp example for C++ applications
   cp -r ${HIPVS_SRC_DIR}/examples/cpp ${HIPVS_SRC_DIR}/examples/cmake ${PROJECT_SRC_DIR}

An existing project can also be modified to use hipVS by copying the contents in the
``configure rapids-cmake`` and ``configure cuvs`` sections of the provided ``CMakeLists.txt`` into the
existing project, along with cmake scripts in ``<hipvs_source>/examples/cmake``.

To build against a hipVS version other than the latest release, set the CMake variable
``CUVS_PINNED_TAG``, in ``get_cuvs.cmake``, to the branch to build against. Alternatively, the
CMake variable ``CPM_cuvs_SOURCE`` can be set to the path to a local directory containing
the hipVS source code, possibly with custom code changes that you want to test.

Make sure to link against the appropriate CMake targets.
Use ``cuvs::c_api`` and ``cuvs::cuvs`` to use the C and C++ shared libraries respectively.

.. code-block:: cmake

   target_link_libraries(your_app_target PRIVATE cuvs::cuvs)

hipVS Examples
--------------

These examples demonstrate various k-nearest neighbors vector search algorithms available in hipVS.
At a high level, each example performs the following steps:

1. Synthetic vector datasets and queries are generated using the random functionality of the
   `RAFT <https://rocm.docs.amd.com/projects/hipRaft/en/latest/>`_ library.
2. A search index is built on the dataset, which accelerates subsequent searches.
3. The search algorithm is run to find the *k* nearest neighbors of the vectors in the query set.
4. The results—consisting of the indices of the nearest dataset vectors and their distances to the query
   vectors—are displayed on the screen.

The following sections provide more details about the specific algorithms used.

C++ Examples
~~~~~~~~~~~~

Brute-Force Search with Bitmap Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates the :doc:`Brute Force <../reference/cpp_api/neighbors_bruteforce>`
search algorithm. The index is built using :ref:`neighbors::brute_force::build <neighbors-bruteforce-index-build>`
function, and the search is performed using the :ref:`neighbors::brute_force::search <neighbors-bruteforce-index-search>`
function. It also demonstrates how to use the ``bitmap_filter`` parameter to specify vectors in the ``search`` dataset to
be included or excluded from the search.

CAGRA Approximate Nearest Neighbors Search Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example showcases the :doc:`CAGRA <../reference/cpp_api/neighbors_cagra>` approximate nearest
neighbors search algorithm. This algorithm was specifically developed for GPUs. The search index is
built using the :ref:`neighbors::cagra::build <neighbors-cagra-index-build>` function, and the search is
performed using the :ref:`neighbors::cagra::search <neighbors-cagra-index-search>` function.

CAGRA Persistent Kernels Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example explores advanced use cases with the :doc:`CAGRA <../reference/cpp_api/neighbors_cagra>`
algorithm. The search is run in three different modes, and their performance is measured for comparison.
The query set is split into two batches to verify performance consistency across similarly sized groups.

1. **Full batch search**: The algorithm runs the entire batch at once.
2. **Asynchronous search**: Queries are launched in parallel, limited by ``MaxJobSize``.
3. **Persistent kernel search**: Similar to the second method, but the search kernels are made
   persistent on the device, reducing kernel launch overhead.

The execution times for all three methods across both batches are printed to the screen.

Dynamic Batching Example
^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates :doc:`Dynamic Batching <../reference/cpp_api/neighbors_dynamic_batching>` in search execution.
Similar to the persistent kernel example, the queries are split into two batches, and three methods are applied:

1. Full batch search
2. Asynchronous search
3. **Dynamic batching**: A dispatch timeout is set, and :ref:`neighbors::dynamic_batching::search <neighbors-dynamic-batching-index-search>`
   is used. It waits to collect smaller queries before the timeout, and runs them together,
   reducing kernel launch overhead.

IVF Flat Example
^^^^^^^^^^^^^^^^

This example demonstrates the :doc:`IVF (inverted file index) search algorithm, using flat vectors <../reference/cpp_api/neighbors_ivf_flat>`.
Two search methods are shown:

1. Building the index from the full dataset
2. Building the index from a sub-sampled dataset

The index built from the sub-sampled dataset is extended, using :ref:`cuvs::neighbors::ivf_flat::extend <neighbors-ivf-flat-index-extend>`,
with the entire dataset since
IVF stores all input vectors in the index. Results from both methods are printed to the screen.

IVF PQ (Product Quantization) Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building on the IVF flat example, this version demonstrates the :doc:`IVF PQ <../reference/cpp_api/neighbors_ivf_pq>`
algorithm, which stores quantized vector values to save memory. The example uses 16-bit float
precision for quantization. This example also demonstrates the :doc:`cuvs::neighbors::refinement::refine <../reference/cpp_api/neighbors_refine>`
functionality, by using IVF PQ to search for *k+n* vectors and refining the results to *k* vectors
based on their actual distance, computed using full precision. Results from both the base and
refined searches are shown.

Vamana Example
^^^^^^^^^^^^^^

:doc:`Vamana <../reference/cpp_api/neighbors_vamana>` is the algorithm used to build the index for the diskANN vector search algorithm.
This example demonstrates the GPU-optimized implementation of the index computation in hipVS.

The input dataset is provided as a binary file of vectors. The expected format is:

1. 4 bytes unsigned int value giving the number of vectors in the dataset, (``N``).
2. 4 bytes unsigned int value giving the dimension of the vectors, (``dim``).
3. ``N * dim`` values in row-major order: use 32-bit floating point when ``datatype`` is ``float``,
   or 8-bit signed integers when ``datatype`` is ``int8``.

The index is built and saved to an output file. This index file can then be used with the open source
diskANN implementation to search for *kNN* vectors.

The Vamana algorithm can be tuned using command-line arguments as described below:

.. code-block:: text

   Usage: ./VAMANA_EXAMPLE <data filename> <output filename> <datatype> <graph degree> <visited_size> <max_fraction> <iterations> [<codebook prefix>]
   <datatype> must be float or int8 (must match the payload type after the header above).
   Graph degree sizes supported: 32, 64, 128, 256 (must be greater than or equal to the device warp/wavefront size)
   Visited_size must be > degree and a power of 2.
   max_fraction > 0 and <= 1. Typical values are 0.06 or 0.1.
   Default iterations = 1.0, increase for better quality graph.
   Optional <codebook prefix>: path prefix for PQ pivots and rotation matrix files
   (${codebook_prefix}_pq_pivots.bin and ${codebook_prefix}_pq_pivots.bin_rotation_matrix.bin).

C Examples
~~~~~~~~~~

Most of the C examples are ports of the corresponding C++ examples. One thing to note is that
the C binding of the hipVS library uses the `DLPack <https://dmlc.github.io/dlpack/latest/>`_
library for representing tensor layouts. Following are the C examples:

CAGRA Example
^^^^^^^^^^^^^

A basic port of the C++ CAGRA example.

IVF Flat Example
^^^^^^^^^^^^^^^^

A basic port of the C++ IVF Flat example.

IVF PQ Example
^^^^^^^^^^^^^^

A basic port of the C++ IVF PQ example.

L2 Example
^^^^^^^^^^

This example is different from the others in that it demonstrates the GPU optimized vector
distance computation functionality in hipVS. This example computes the L2 distance between two
vectors using the :ref:`cuvsPairwiseDistance <pairwise-distance-c>` function and prints the result to the screen.

Python Examples
~~~~~~~~~~~~~~~

Similar to the C and C++ examples, these examples demonstrate the use of kNN and ANN algorithms
to do vector search. The examples are:

1. Cagra Example: Computes k nearest neighbors on randomly generated dataset and query
   vectors using the Cagra algorithm.
2. IVF Flat Example: Computes k nearest neighbors on randomly generated dataset and query
   vectors using the IVF-Flat algorithm. Also demonstrates index building with explicitly specified
   training set and then extending the index with the full input dataset for searching.
3. IVF PQ Example: Computes k nearest neighbors on randomly generated dataset and query
   vectors using the IVF-PQ algorithm. First k+n neighbors are found using the quantized search,
   which are then refined to k vectors using full precision distance computation.
4. Pairwise L2 Distances Example: Computes the pairwise l2 distance between 2 sets of vectors

To run these examples, you need a Python environment with hipVS and its dependencies installed.
Please follow the :ref:`Python library <python-library>` documentation
to see the list of dependencies and instructions on setting
up a conda environment for hipVS. Inside the environment, the examples can be run using
Python. For example:

.. code-block:: bash

   $ python3 examples/python/cagra_example.py

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

Some demos in the form of Jupyter notebooks are available under the ``hipVS/notebooks`` folder.
The python package dependencies for these notebooks can be installed using
the ``hipVS/notebooks/requirements.txt`` file.

.. code-block:: bash

   $ pip install -r requirements.txt

Before running the notebooks, make sure to set the ROCM_HOME environment variable to the path of your ROCm installation.

To run the notebooks interactively, start the Jupyter server:

.. code-block:: bash

   $ cd hipVS/notebooks
   $ jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

After starting the server, open your browser and navigate to the URL printed in the console,
which will be in the form: ``http://127.0.0.1:8888/tree?token=...``
