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

.. _build-hipvs:

**************************
Building hipVS from source
**************************

hipVS currently provides C++, C, Python and Rust APIs. The following instructions provide steps to build and test hipVS from source files provided in the `hipVS repository <https://github.com/ROCm-DS/hipVS>`_.

Tested GPUs
===========

.. list-table::
   :header-rows: 1

   * - AMD Instinct GPU
     - Architecture
     - Wavefront Size
     - LLVM target
   * - MI355X
     - CDNA4
     - 64
     - gfx950
   * - MI350X
     - CDNA4
     - 64
     - gfx950
   * - MI325X
     - CDNA3
     - 64
     - gfx942
   * - MI300X
     - CDNA3
     - 64
     - gfx942
   * - MI300A
     - CDNA3
     - 64
     - gfx942
   * - MI250X
     - CDNA2
     - 64
     - gfx90a
   * - MI250
     - CDNA2
     - 64
     - gfx90a
   * - MI210
     - CDNA2
     - 64
     - gfx90a


Dependencies
============

hipVS builds against the AMD ROCm software stack, that is, the ROCm runtime, HIP compiler tool-chain, and a GPU driver that matches your ROCm version.

Install ROCm 7.2.0 or later (at minimum, a version that supports the GPUs listed above) and verify that the ``rocminfo`` and ``hipcc`` commands are available in your ``PATH``. For installation instructions, see `ROCm Installation <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/>`__.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Name
     - Version / Notes
   * - `cmake <https://cmake.org/>`_
     - ≥ 3.31.0
   * - `ninja <https://ninja-build.org/>`_
     - ≥ 1.11.1
   * - `hipsolver <https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - `hipblas <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - `hipblaslt <https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - `hiprand <https://rocm.docs.amd.com/projects/hipRAND/en/latest/>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - `hipsparse <https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - `libblas-dev <https://www.netlib.org/lapack/>`_
     - Tested with 3.12.0
   * - `liblapack-dev <https://www.netlib.org/lapack/>`_
     - Tested with 3.12.0
   * - `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_
     - Tested with 7.6.1
   * - `libopenblas-dev <https://github.com/OpenMathLib/OpenBLAS>`_
     - Tested with 0.3.26
   * - **Additional Required Dependencies**
     -
   * - **\*** `hipMM <https://github.com/ROCm-DS/hipMM>`_
     - 4.0.0
   * - **\*** `hipCollections <https://github.com/ROCm/hipCollections>`_
     - 0.4.0
   * - **\*** `hipRAFT <https://github.com/ROCm-DS/hipRAFT>`_
     - 1.0.0
   * - **\*** `hipCUB <https://github.com/ROCm/hipCUB>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - **\*** `rocThrust <https://github.com/ROCm/rocThrust>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - **\*\*** `OpenMP <https://www.openmp.org/>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - **Optional Dependencies**
     -
   * - `RCCL <https://github.com/ROCm/rccl>`_
     - Version that comes bundled with ROCm ≥ 7.2.0
   * - `UCX <https://github.com/openucx/ucx>`_
     - ≥ 1.17.0
   * - `Googletest <https://github.com/google/googletest>`_
     - ≥ 1.13.0
   * - `Googlebench <https://github.com/google/benchmark>`_
     - ≥ 1.13.0
   * - `Doxygen <https://github.com/doxygen/doxygen>`_
     - ≥ 1.8.20

.. note::

   ``*`` - If not found locally the CMake build system will attempt to download a compatible version using `ROCmDS-cmake <https://github.com/ROCm-DS/ROCmDS-cmake>`_.

   ``**`` - The ``OpenMP`` toolchain is automatically installed as part of the standard ROCm installation and is available under ``/opt/rocm-{version}/llvm``.

Docker
------

hipVS also provides a ``Dockerfile`` that encapsulates all the above dependencies for development. Below are the instructions to create a container using this Dockerfile.

.. code-block:: bash

   cd <REPO_ROOT>
   # If BuildKit is not available.
   DOCKER_BUILDKIT=1 docker build -t <HIPVS_DEV_IMAGE> .
   # If BuildKit is available: "docker buildx build  -t  <HIPVS_DEV_IMAGE> ."
   docker run -d -it --cap-add=SYS_PTRACE \
          --device=/dev/kfd \
          --device=/dev/dri \
          --group-add=video \
          --ipc=host \
          --name <CONTAINER_NAME> \
          --init \
          --network=host \
          --security-opt seccomp=unconfined \
          -v <REPO_ROOT>:<REPO_ROOT> <HIPVS_DEV_IMAGE>  tail -f /dev/null
   docker exec -it <CONTAINER_NAME> bash

This container will have all necessary packages installed to build and run hipVS properly. The preceding command will create an Ubuntu 24.04 container that has ROCm installed. If you wish to use Ubuntu 22.04, replace the docker build command with the following:

.. code-block:: bash

   docker build --build-arg UBUNTU=22.04 -t <HIPVS_DEV_IMAGE> .

Environment variables
=====================

.. code-block:: bash

   export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake # Set CMAKE_PREFIX_PATH to point to the ROCm installation site

Build Topics
============

The rest of the *Building hipVS* topic is broken down as follows: 

* :ref:`cpp-library`
* :ref:`python-library`
* :ref:`rust-library`
* :ref:`packaging`
* :ref:`build-docs`
* :ref:`build-benchmarks`

.. _cpp-library:

C and C++ library
=================

The core functionality of hipVS is implemented in HIP/C++ and its functionality is exposed through ``libcuvs.so``. There is also the C library ``libcuvs_c.so`` which is a wrapper around the C++ library. The C library is used to provide a C API to the functionality of the C++ library.

Building and installing using build.sh
---------------------------------------

A utility script ``<HIPVS_ROOT>/build.sh`` is provided and is the entry-point to building various components.

``build.sh`` uses `ROCmDS-cmake <https://github.com/ROCm-DS/ROCmDS-cmake>`_, which will automatically download any dependencies that are not already installed.

The following will download the required dependencies, build and install the hipVS CMake package to the configured CMake install prefix.

.. code-block:: bash

   cd <HIPVS_ROOT>
   ./build.sh libcuvs

The installation destination directory follows a similar structure:

.. code-block:: text

   <HIPVS_INSTALLATION_ROOT>
   |-- include # Public C and C++ headers
   |   |--*
   `-- lib
       |-- cmake
       |   |-- cuvs
       |   |   |-- cuvs-c_api-c-targets-release.cmake
       |   |   |-- cuvs-c_api-c-targets.cmake
       |   |   |-- cuvs-config-version.cmake
       |   |   |-- cuvs-config.cmake
       |   |   |-- cuvs-dependencies.cmake
       |   |   |-- cuvs-targets-release.cmake
       |   |   `-- cuvs-targets.cmake
       |   `-- hnswlib
       |       |-- hnswlib-config-version.cmake
       |       |-- hnswlib-config.cmake
       |       `-- hnswlib-targets.cmake
       |-- libcuvs.so        # C++ Shared Library
       |-- libcuvs_c.so      # C   Shared Library
       `-- libcuvs_static.a  # C++ Static Library

C and C++ tests
---------------

Compile the tests using the ``tests`` argument passed to ``build.sh``.

.. code-block:: bash

   ./build.sh tests

The tests are broken into algorithm categories, so you'll find several binaries in ``cpp/build/gtests`` named ``*_TEST``.

.. code-block:: text

   <HIPVS_ROOT>/cpp/build/gtests/
   |-- BRUTEFORCE_C_TEST
   |-- CAGRA_C_TEST
   |-- CLUSTER_TEST
   |-- DISTANCE_C_TEST
   |-- DISTANCE_TEST
   |-- HNSW_C_TEST
   |-- INTEROP_TEST
   |-- IVF_FLAT_C_TEST
   |-- IVF_PQ_C_TEST
   |-- NEIGHBORS_ANN_BRUTE_FORCE_TEST
   |-- NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST
   |-- NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST
   |-- NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST
   |-- NEIGHBORS_ANN_CAGRA_TEST_BUGS
   |-- NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST
   |-- NEIGHBORS_ANN_IVF_FLAT_TEST
   |-- NEIGHBORS_ANN_IVF_PQ_TEST
   |-- NEIGHBORS_ANN_NN_DESCENT_TEST
   |-- NEIGHBORS_ANN_VAMANA_TEST
   |-- NEIGHBORS_DYNAMIC_BATCHING_TEST
   |-- NEIGHBORS_HNSW_TEST
   |-- NEIGHBORS_TEST
   |-- PREPROCESSING_TEST
   |-- SPARSE_TEST
   |-- STATS_TEST
   |-- cuvs_c_neighbors_test
   |-- cuvs_c_test

For example, to run the distance module tests:

.. code-block:: bash

   ./cpp/build/gtests/DISTANCE_TEST

It can take significant time to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the ``--limit-tests`` option in ``build.sh``:

.. code-block:: bash

   ./build.sh tests -n --limit-tests="cuvs_c_test;STATS_TEST"

Running all the tests using ctest:

.. code-block:: bash

   cd <HIPVS_ROOT>/cpp/build/
   ctest --test-dir ./tests # If "--limit-tests" is specified, only a subset of tests are built. To run all the tests remove "--limit-tests" when building through `build.sh`

ccache and sccache
------------------

`ccache <https://ccache.dev/>`_ and `sccache <https://github.com/mozilla/sccache>`_ can be used to better cache parts of the build when rebuilding frequently, such as when working on a new feature. You can also use ``ccache`` or ``sccache`` with ``build.sh``:

.. code-block:: bash

   ./build.sh libcuvs --cache-tool=ccache

GPU architecture selection
---------------------------

--allgpuarch
~~~~~~~~~~~~

Builds hipVS for all supported GPU architectures, increasing portability but also build time. You can also use --allgpuarch with ``build.sh``:

.. code-block:: bash

   ./build.sh clean
   ./build.sh libcuvs tests --allgpuarch

.. note::

   Always execute a clean build or manually delete the build directory before running this argument if a previous build is present, to prevent potential build conflicts or errors.

Compile only for a specified GPU architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When specifying target architectures, provide them as a single string enclosed in double quotes. For multiple architectures, separate each with a semicolon (;).

.. code-block:: bash

   ./build.sh libcuvs tests --gpu-arch="gfx90a;gfx942"

OR

.. code-block:: bash

   ./build.sh libcuvs tests --gpu-arch="gfx942"

.. note::

   Do not specify both ``--gpu-arch`` and ``--allgpuarch`` flags in the same build command. Avoid using multiple separate ``--gpu-arch`` flags by combining all target architectures into one ``--gpu-arch`` option.

Using CMake directly
--------------------

The ``build.sh`` script wraps common CMake configuration options. For finer control, invoke ``cmake`` directly as shown below.

The ``CMAKE_INSTALL_PREFIX`` option instructs CMake to install ``hipVS`` into a specific location.

.. code-block:: bash

   cd <HIPVS_ROOT>/cpp
   mkdir -p build && rm -rf build/*
   cd build
   # Configuration stage
   cmake -S .. \
         -G Ninja \
         -B . \
         -DCMAKE_INSTALL_PREFIX=install \
         -DCMAKE_HIP_ARCHITECTURES=NATIVE \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TESTS=OFF \
         -DCMAKE_CXX_COMPILER=hipcc
   # Build the cuvs target
   ninja cuvs
   # Install
   ninja install

.. _python-library:

Python Library
==============

hipVS Python packages have a dependency on ``CuPy``, which requires the ``ROCM_HOME`` environment variable to be set to the base folder of the ROCm installation, typically ``/opt/rocm``.

Build and install hipVS python packages
----------------------------------------

Step 1: Setup Conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conda environment scripts are provided for installing the necessary dependencies to build the Python libraries from source. It is preferred to use `micromamba <https://github.com/mamba-org/mamba>`_ as it's a fully statically-linked, self-contained, executable. ``micromamba`` can be installed by following the instructions on this page: `Micromamba Installation <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_.

.. code-block:: bash

   cd <HIPVS_ROOT>
   micromamba  env create --name hipvs  --file conda/environments/all_rocm_arch-x86_64.yaml
   # To initialize the current bash shell, run:
   eval "$(micromamba shell hook --shell bash)"
   micromamba activate hipvs

It is recommended to build the python wheels in a conda environment built from ``all_rocm_arch-x86_64.yaml``. It is also possible to use ``venv`` but it is up to the user to install all the required packages in the environment.

Step 2: Building and installing hipVS python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using build.sh
^^^^^^^^^^^^^^

The Python libraries can be built and installed using the build.sh script:

.. code-block:: bash

   # From within the hipvs environment:
   cd <HIPVS_ROOT>
   # Build and install hipvs-python and required dependencies.
   ./build.sh libcuvs python
   # Test installation
   python
   >> import cuvs # Import should succeed

Building and installing the wheels manually
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Install the cuvs CMake package in the hipvs conda environment
   cd <HIPVS_ROOT>
   ./build.sh libcuvs

   # Build and install the libcuvs python wheel
   cd <HIPVS_ROOT>/python/libcuvs
   pip install -v --no-build-isolation  .

   # Build and install the cuvs python wheel
   cd <HIPVS_ROOT>/python/cuvs/
   pip install -v --no-build-isolation  .
   # Test installation
   python
   >> import cuvs

Running the python tests
-------------------------

.. code-block:: bash

   #set ROCM_HOME path
   export ROCM_HOME=<path to rcom home> (e.g /opt/rocm)

   # From within the hipvs environment
   cd <HIPVS_ROOT>/python/cuvs/
   py.test -v -s

.. _rust-library:

Rust Library
============

Building and running the Rust bindings tests assumes that Cargo (the Rust package manager and build tool) is installed and available in your environment.
If Cargo is not installed, see the `Rustup documentation <https://rustup.rs/>`_ for installation instructions.

The Rust library can be built and installed using the ``build.sh`` script. As a prerequisite, the ``hipvs-rust`` library depends on the ``hipvs`` C++ library(``libcuvs.so``) and C library(``libcuvs_c.so``).
The ``hipvs`` C++ library must be built and installed before building the Rust library. If using the ``build.sh`` script, the following command can be used to build and install the hipVS C++ and Rust libraries:

.. code-block:: bash

   # From within a hipvs conda environment:
   cd <HIPVS_ROOT>
   # The following command will first build and install the hipvs C++ library into the conda environment, then build and install the hipvs rust library and example.
   # It will also run the rust tests after building the library.
   ./build.sh libcuvs rust

.. note::

   The Rust option invoked through ``build.sh`` will not only build the ``hipvs`` and ``hipvs-sys`` crates, but also run the tests after.

Running the Rust example after building through build.sh
---------------------------------------------------------

.. code-block:: bash

   # From within the hipvs environment
   LD_LIBRARY_PATH=$CONDA_PREFIX/lib <HIPVS_ROOT>/rust/target/debug/examples/cagra

.. _packaging:

Packaging
=========

Packaging with build.sh
------------------------

The following command will generate a debian package: ``hipvs_<VERSION>_amd64.deb`` in ``<HIPVS_ROOT>/cpp/build``.

.. code-block:: bash

   ./build.sh libcuvs package

Custom cpack generators
-----------------------

To generate other types of packages such as ``tar`` or ``RPM`` packages:

.. code-block:: bash

   # Configure
   mkdir -p <HIPVS_ROOT>/cpp/build
   rm -rf <HIPVS_ROOT>/cpp/build/*
   cd <HIPVS_ROOT>/cpp/build
   cmake -S .. \
         -G Ninja \
         -B . \
         -DCMAKE_INSTALL_PREFIX=install \
         -DCMAKE_HIP_ARCHITECTURES=NATIVE \
         -DCMAKE_BUILD_TYPE=Release \
         -DRAFT_COMPILE_LIBRARY=ON \
         -DBUILD_TESTS=OFF \
         -DCMAKE_CXX_COMPILER=hipcc

   # Install to staging area
   ninja install

   # Invoke cpack to generate package
   cpack -G RPM # To generate a RPM package. hipvs-<version>-Linux.rpm will be created at <HIPVS_ROOT>/cpp/build.
   cpack -G TGZ # To generate a TGZ package. hipvs-<version>-Linux.tar.gz will be created at <HIPVS_ROOT>/cpp/build.

.. _build-docs:

Building Documentation
======================

Prepare the environment to build documentation using the following commands:

.. code-block:: bash

   cd <HIPVS_ROOT>
   # Activate the hipvs conda environment.
   micromamba activate hipvs
   # Install dependencies and tools required for generating documentation.
   pip install -r docs_amd/sphinx/requirements.txt

Use build.sh to generate documentation
---------------------------------------

.. code-block:: bash

   cd <HIPVS_ROOT>
   ./build.sh docs clean

Navigate to ``<HIPVS_ROOT>/docs/_build`` and use the tool of your choice, e.g. Firefox, to open and examine the root level html file, ``index.html``. From this point you should be able to navigate through the documentation.

.. _build-benchmarks:

Python Benchmark Framework
===========================

hipVS ships a Python-based ANN (Approximate Nearest Neighbor) benchmark framework (``hipvs-bench``) that makes it straightforward to compare index build times, search throughput, and search latency across multiple algorithms and datasets. The Python layer drives C++ benchmark binaries and produces CSV result files and plots.

Building the C++ benchmark binaries
------------------------------------

The benchmark executables are built separately from the main library. From within the hipvs conda environment:

.. code-block:: bash

   cd <HIPVS_ROOT>
   ./build.sh bench-ann

To build only a specific subset of algorithms, pass a semicolon-separated list of executable names to ``--limit-bench-ann``:

.. code-block:: bash

   ./build.sh bench-ann -n --limit-bench-ann="CUVS_CAGRA_ANN_BENCH;CUVS_IVF_PQ_ANN_BENCH;HNSWLIB_ANN_BENCH"

Available targets include:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Target
     - Algorithm
     - Supported on ROCm
   * - ``CUVS_CAGRA_ANN_BENCH``
     - hipVS CAGRA
     - Yes
   * - ``CUVS_IVF_FLAT_ANN_BENCH``
     - hipVS IVF-Flat
     - Yes
   * - ``CUVS_IVF_PQ_ANN_BENCH``
     - hipVS IVF-PQ
     - Yes
   * - ``CUVS_BRUTE_FORCE_ANN_BENCH``
     - hipVS Brute-Force
     - Yes
   * - ``CUVS_CAGRA_HNSWLIB_ANN_BENCH``
     - hipVS CAGRA (HNSWLib search)
     - Yes
   * - ``HNSWLIB_ANN_BENCH``
     - HNSWLib (CPU)
     - Yes
   * - ``FAISS_GPU_IVF_FLAT_ANN_BENCH``
     - FAISS GPU IVF-Flat
     - Yes
   * - ``FAISS_GPU_IVF_PQ_ANN_BENCH``
     - FAISS GPU IVF-PQ
     - Yes
   * - ``FAISS_GPU_FLAT_ANN_BENCH``
     - FAISS GPU Flat
     - Yes
   * - ``FAISS_CPU_IVF_FLAT_ANN_BENCH``
     - FAISS CPU IVF-Flat
     - Yes
   * - ``FAISS_CPU_IVF_PQ_ANN_BENCH``
     - FAISS CPU IVF-PQ
     - Yes
   * - ``FAISS_CPU_FLAT_ANN_BENCH``
     - FAISS CPU Flat
     - Yes
   * - ``FAISS_CPU_HNSW_FLAT_ANN_BENCH``
     - FAISS CPU HNSW
     - Yes
   * - ``GGNN_ANN_BENCH``
     - GGNN
     - No — requires hipification and upstream build system changes
   * - ``DISKANN_MEMORY_ANN_BENCH``
     - DiskANN (in-memory)
     - No — not yet ported to ROCm
   * - ``DISKANN_SSD_ANN_BENCH``
     - DiskANN (SSD)
     - No — not yet ported to ROCm
   * - ``CUVS_VAMANA_ANN_BENCH``
     - hipVS Vamana
     - No — depends on DiskANN
   * - ``CUVS_CAGRA_DISKANN_ANN_BENCH``
     - hipVS CAGRA + DiskANN search
     - No — depends on DiskANN

The compiled binaries are placed under ``<HIPVS_ROOT>/cpp/build/bench``.

Installing the Python benchmark package
----------------------------------------

After activating the hipvs conda environment and building ``libcuvs``, install the ``hipvs-bench`` Python package:

.. code-block:: bash

   # From within the hipvs environment
   cd <HIPVS_ROOT>/python/cuvs_bench
   pip install -v --no-build-isolation  .

Downloading datasets
---------------------

The benchmark framework reads datasets in a binary format (``.fbin``, ``.ibin``, etc.). Datasets are stored under the path specified by the ``RAPIDS_DATASET_ROOT_DIR`` environment variable, or a ``datasets/`` sub-folder in the current working directory if the variable is not set.

.. code-block:: bash

   export RAPIDS_DATASET_ROOT_DIR=/path/to/datasets

Million-scale datasets
~~~~~~~~~~~~~~~~~~~~~~~

Million-scale datasets can be downloaded automatically from `ann-benchmarks.com <http://ann-benchmarks.com>`_ using the ``get_dataset`` module. The downloaded HDF5 files are converted to the binary format expected by the benchmark binaries.

.. code-block:: bash

   # Example: download and prepare the SIFT-128 Euclidean dataset
   python -m cuvs_bench.get_dataset --dataset sift-128-euclidean

   # For angular (cosine) datasets, pass --normalize to convert cosine to inner product
   python -m cuvs_bench.get_dataset --dataset glove-100-angular --normalize

The following million-scale datasets are available:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Vectors
     - Dimensions
     - Distance
   * - ``deep-image-96-angular``
     - 10M
     - 96
     - Angular
   * - ``fashion-mnist-784-euclidean``
     - 60K
     - 784
     - Euclidean
   * - ``glove-50-angular``
     - 1.1M
     - 50
     - Angular
   * - ``glove-100-angular``
     - 1.1M
     - 100
     - Angular
   * - ``mnist-784-euclidean``
     - 60K
     - 784
     - Euclidean
   * - ``nytimes-256-angular``
     - 290K
     - 256
     - Angular
   * - ``sift-128-euclidean``
     - 1M
     - 128
     - Euclidean

All datasets above include ground truth for the top-100 nearest neighbors, so ``k`` must be ≤ 100.

.. note::

   Angular datasets are renamed from ``*-angular`` to ``*-inner`` after normalization. Use the ``-inner`` suffix when specifying ``--dataset`` for ``cuvs_bench.run`` and ``cuvs_bench.plot``.

Billion-scale datasets
~~~~~~~~~~~~~~~~~~~~~~~

Billion-scale datasets must be downloaded manually from `big-ann-benchmarks.com <http://big-ann-benchmarks.com/neurips21.html>`_. After downloading, the ground truth file (which interleaves neighbors and distances) must be split into two separate files:

.. code-block:: bash

   mkdir -p datasets/deep-1B
   # Download the "Ground Truth" file for the Yandex DEEP dataset from big-ann-benchmarks.com
   # (file is named deep_new_groundtruth.public.10K.bin)
   python -m cuvs_bench.split_groundtruth --groundtruth datasets/deep-1B/deep_new_groundtruth.public.10K.bin
   # Produces: groundtruth.neighbors.ibin and groundtruth.distances.fbin

Converting HDF5 files manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have an HDF5 dataset from ann-benchmarks.com that you want to convert manually:

.. code-block:: bash

   # Without normalization (Euclidean datasets)
   python python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py <input>.hdf5

   # With normalization (Angular/cosine datasets)
   python python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py -n <input>.hdf5

This produces four binary files: ``<input>.base.fbin``, ``<input>.query.fbin``, ``<input>.groundtruth.neighbors.ibin``, and ``<input>.groundtruth.distances.fbin``.

To convert a dataset from ``float32`` to ``float16`` format (for algorithms that benefit from it):

.. code-block:: bash

   python python/cuvs_bench/cuvs_bench/get_dataset/fbin_to_f16bin.py <base>.fbin

Generating ground truth for custom datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a custom dataset without a ground truth file, you can generate one using the ``generate_groundtruth`` module:

.. code-block:: bash

   # With an existing query file
   python -m cuvs_bench.generate_groundtruth \
       --dataset /path/to/base.fbin \
       --output groundtruth_dir \
       --queries /path/to/query.fbin

   # With randomly generated queries
   python -m cuvs_bench.generate_groundtruth \
       --dataset /path/to/base.fbin \
       --output groundtruth_dir \
       --queries random \
       --n_queries 10000

Running benchmarks
------------------

The ``cuvs_bench.run`` module drives the C++ benchmark binaries. By default it runs both the build (index construction) and search phases and exports results to CSV.

End-to-end example (smaller scale)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Step 1: Download dataset
   python -m cuvs_bench.get_dataset --dataset sift-128-euclidean

   # Step 2: Build index and run search
   python -m cuvs_bench.run \
       --dataset sift-128-euclidean \
       --algorithms cuvs_cagra \
       --batch-size 10000 \
       -k 10

   # Step 3: Export intermediate JSON results to CSV (done automatically, but can be run manually)
   python -m cuvs_bench.run --data-export --dataset sift-128-euclidean

   # Step 4: Plot results
   python -m cuvs_bench.plot --dataset sift-128-euclidean

Selecting algorithms
~~~~~~~~~~~~~~~~~~~~

Multiple algorithms can be benchmarked simultaneously by providing a comma-separated list. For example, to compare hipVS CAGRA, IVF-PQ, and HNSWLib:

.. code-block:: bash

   python -m cuvs_bench.run \
       --dataset glove-100-inner \
       --algorithms cuvs_cagra,cuvs_ivf_pq,hnswlib \
       --batch-size 10000 \
       -k 10

Supported algorithm names:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Name
     - Description
     - Supported on ROCm
   * - ``cuvs_cagra``
     - hipVS CAGRA graph-based index
     - Yes
   * - ``cuvs_ivf_flat``
     - hipVS IVF-Flat
     - Yes
   * - ``cuvs_ivf_pq``
     - hipVS IVF-PQ
     - Yes
   * - ``cuvs_brute_force``
     - hipVS brute-force search
     - Yes
   * - ``cuvs_cagra_hnswlib``
     - CAGRA index searched with HNSWLib
     - Yes
   * - ``hnswlib``
     - HNSWLib (CPU)
     - Yes
   * - ``faiss_gpu_flat``
     - FAISS GPU flat
     - Yes
   * - ``faiss_gpu_ivf_flat``
     - FAISS GPU IVF-Flat
     - Yes
   * - ``faiss_gpu_ivf_pq``
     - FAISS GPU IVF-PQ
     - Yes
   * - ``faiss_cpu_flat``
     - FAISS CPU flat
     - Yes
   * - ``faiss_cpu_ivf_flat``
     - FAISS CPU IVF-Flat
     - Yes
   * - ``faiss_cpu_ivf_pq``
     - FAISS CPU IVF-PQ
     - Yes
   * - ``faiss_cpu_hnsw_flat``
     - FAISS CPU HNSW
     - Yes
   * - ``cuvs_vamana``
     - hipVS Vamana (DiskANN)
     - No — depends on DiskANN
   * - ``diskann_memory``
     - DiskANN in-memory search
     - No — not yet ported to ROCm
   * - ``diskann_ssd``
     - DiskANN SSD-based search
     - No — not yet ported to ROCm
   * - ``ggnn``
     - GGNN
     - No — requires hipification

Latency vs. throughput mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, benchmarks run in **latency** mode (each batch is timed individually). To measure **throughput** (queries per second with multiple concurrent threads):

.. code-block:: bash

   python -m cuvs_bench.run \
       --dataset glove-100-inner \
       --algorithms cuvs_cagra \
       --search-mode throughput \
       --search-threads 1:4 \
       -k 10

``--search-threads`` accepts a single value or a ``min:max`` range. Powers of 2 between ``min`` and ``max`` are tested.

Running only build or search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build the index only
   python -m cuvs_bench.run --dataset sift-128-euclidean --algorithms cuvs_cagra --build -k 10

   # Run search only (index must already be built)
   python -m cuvs_bench.run --dataset sift-128-euclidean --algorithms cuvs_cagra --search -k 10

Dry run
~~~~~~~

To preview the benchmark commands without executing them:

.. code-block:: bash

   python -m cuvs_bench.run --dataset sift-128-euclidean --algorithms cuvs_cagra --dry-run -k 10

Plotting results
----------------

After running benchmarks, use ``cuvs_bench.plot`` to generate Recall vs. Latency and Build Time plots:

.. code-block:: bash

   # Plot both search (Recall vs Latency) and build time charts
   python -m cuvs_bench.plot --dataset sift-128-euclidean

   # Plot throughput instead of latency
   python -m cuvs_bench.plot --dataset sift-128-euclidean --mode throughput

   # Specify output directory for PNG files
   python -m cuvs_bench.plot --dataset sift-128-euclidean --output-filepath /path/to/output

The plot command saves two PNG files per dataset:

- ``search-<dataset>-k<k>-batch_size<bs>.png`` — Recall vs. Latency (or QPS) Pareto frontier
- ``build-<dataset>-k<k>-batch_size<bs>.png`` — Average build time at different recall thresholds
