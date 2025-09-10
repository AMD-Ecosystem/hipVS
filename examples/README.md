# Introduction

Examples demonstrating the use of the hipVS library are provided in the `examples` folder.
Currently, examples targeting C, C++, and Python bindings are available.  These
example projects can be used as templates to build your own application using
hipVS, or to add hipVS to existing projects.

# Building the examples

To build the examples, use the provided `build.sh` script. This is a bash script
that calls the appropriate CMake commands, so you can look into it to see the typical
CMake based build workflow.

The dependencies of the examples project, including hipVS, are automatically fetched and built
by the CMake scripts, but there are some basic pre-requisites and dependencies that need to be
satisfied for a successful build. Please refer to the
[Build and installation](../docs_amd/build_and_install/build_and_install.md#introduction)
documentation for instructions on setting up your developer environment for building hipVS.

The individual example project directories can be copied as a starting point to build a new
standalone application using hipVS. Make sure to also copy the `<hipvs_source>/examples/cmake`
folder, as it contains the functionality to fetch and build hipVS.

```bash
# project directory for a new hipVS based application
HIPVS_SRC_DIR=...
PROJECT_SRC_DIR=...

# copy the cpp example for C++ applications
cp -r ${HIPVS_SRC_DIR}/examples/cpp ${HIPVS_SRC_DIR}/examples/cmake ${PROJECT_SRC_DIR}
```

An existing project can also be modified to use hipVS by copying the contents in the
`configure rapids-cmake` and `configure cuvs` sections of the provided `CMakeLists.txt` into the
existing project, along with cmake scripts in `<hipvs_source>/examples/cmake`.

To build against a hipVS version other than the latest release, set the CMake variable
`CUVS_PINNED_TAG`, in `get_cuvs.cmake`, to the branch to build against. Alternatively, the
CMake variable `CPM_cuvs_SOURCE` can be set to the path to a local directory containing
the hipVS source code, possibly with custom code changes that you want to test.

Make sure to link against the appropriate CMake targets.
Use `cuvs::c_api` and `cuvs::cuvs` to use the C and C++ shared libraries respectively.

```cmake
target_link_libraries(your_app_target PRIVATE cuvs::cuvs)
```

# hipVS Examples

These examples demonstrate various k-nearest neighbors vector search algorithms available in hipVS.
At a high level, each example performs the following steps:
1. Synthetic vector datasets and queries are generated using the random functionality of the
   [RAFT](https://rocm.docs.amd.com/projects/hipRaft/en/latest/) library.
2. A search index is built on the dataset, which accelerates subsequent searches.
3. The search algorithm is run to find the *k* nearest neighbors of the vectors in the query set.
4. The results—consisting of the indices of the nearest dataset vectors and their distances to the query
   vectors—are displayed on the screen.
The following sections provide more details about the specific algorithms used.

## C++ Examples

### Brute-Force Search with Bitmap Filtering

This example demonstrates the [Brute Force](reference/cpp_api/neighbors_bruteforce:bruteforce)
search algorithm. The index is built using the
[neighbors::brute_force::build](neighbors-bruteforce-index-build)
function, and the search is performed using the [neighbors::brute_force::search](neighbors-bruteforce-index-search)
function. It also demonstrates how to use [neighbors::filtering::bitmap_filter](reference/cpp_api/neighbors_filter:filtering),
which specifies vectors in the input dataset to be excluded from the search.

### CAGRA Approximate Nearest Neighbors Search Algorithm

This example showcases the [CAGRA](reference/cpp_api/neighbors_cagra:cagra) approximate nearest
neighbors search algorithm. This algorithm was specifically developed for GPUs. The search index is
built using the [neighbors::cagra::build](neighbors-cagra-index-build) function, and the search is
performed using the [neighbors::cagra::search](neighbors-cagra-index-search) function.

### CAGRA Persistent Kernels Example

This example explores advanced use cases with the [CAGRA](reference/cpp_api/neighbors_cagra:cagra) algorithm. The search is run in three
different modes, and their performance is measured for comparison. The query set is split into two
batches to verify performance consistency across similarly sized groups.

1. **Full batch search**: The algorithm runs the entire batch at once.
2. **Asynchronous search**: Queries are launched in parallel, limited by `MaxJobSize`.
3. **Persistent kernel search**: Similar to the second method, but the search kernels are made
   persistent on the device, reducing kernel launch overhead.

The execution times for all three methods across both batches are printed to the screen.

### Dynamic Batching Example

This example demonstrates [dynamic batching](neighbors-dynamic-batching) in search execution. Similar to the persistent kernel
example, the queries are split into two batches, and three methods are applied:

1. Full batch search
2. Asynchronous search
3. **Dynamic batching**: A dispatch timeout is set, and [neighbors::dynamic_batching::search](neighbors-dynamic-batching-index-search)
is used. It waits to collect smaller queries before the timeout, and runs them together,
reducing kernel launch overhead.

### IVF Flat Example

This example demonstrates the [IVF (inverted file index) search algorithm, using flat vectors](reference/cpp_api/neighbors_ivf_flat:ivf-flat).
Two search methods are shown:

1. Building the index from the full dataset
2. Building the index from a sub-sampled dataset

The index built from the sub-sampled dataset is extended, using [cuvs::neighbors::ivf_flat::extend](neighbors-ivf-flat-index-extend),
with the entire dataset since
IVF stores all input vectors in the index. Results from both methods are printed to the screen.

### IVF PQ (Product Quantization) Example

Building on the IVF flat example, this version demonstrates the [IVF PQ](reference/cpp_api/neighbors_ivf_pq:ivf-pq)
algorithm, which stores quantized vector values to save memory. The example uses 16-bit float
precision for quantization. This example also demonstrates the [cuvs::neighbors::refine](reference/cpp_api/neighbors_refine:refinement)
functionality, by using IVF PQ to search for *k+n* vectors and refining the results to *k* vectors
based on their actual distance, computed using full precision. Results from both the base and
refined searches are shown.

### Vamana Example

[Vamana](reference/cpp_api/neighbors_vamana:vamana) is the algorithm used to build the index for the diskANN vector search algorithm.
This example demonstrates the GPU-optimized implementation of the index computation in hipVS.

The input dataset is provided as a binary file of vectors. The expected format is:
1. 4 bytes unsigned int value giving the number of vectors in the dataset, (`N`).
2. 4 bytes unsigned int value giving the dimension of the vectors, (`dim`).
3. `N * dim` 32-bit floating point values giving the N vectors.
The index is built and saved to an output file. This index file can then be used with the open source
diskANN implementation to search for *kNN* vectors.

The Vamana algorithm can be tuned using command-line arguments as described below:

```
Usage: ./VAMANA_EXAMPLE <data filename> <output filename> <graph degree> <visited_size> <max_fraction> <iterations>
Input file expected to be binary file of fp32 vectors.
Graph degree sizes supported: 32, 64, 128, 256 (must be greater than or equal to the device warp/wavefront size)
Visited_size must be > degree and a power of 2.
max_fraction > 0 and <= 1. Typical values are 0.06 or 0.1.
Default iterations = 1, increase for better quality graph.
```

## C Examples

Most of the C examples are ports of the corresponding C++ examples. One thing to note is that
the C binding of the hipVS library uses the [DLPack](https://dmlc.github.io/dlpack/latest/)
library for representing tensor layouts. Following are the C examples:

### CAGRA Example

A basic port of the C++ CAGRA example.

### IVF Flat Example

A basic port of the C++ IVF Flat example.

### IVF PQ Example

A basic port of the C++ IVF PQ example.

### L2 Example

This example is different from the others in that it demonstrates the GPU optimized vector
distance computation functionality in hipVS. This example computes the L2 distance between two
vectors using the [cuvsPairwiseDistance](pairwise-distance-c) function and prints the result to the screen.

## Python Examples

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
Please follow this [documentation](../docs_amd/build_and_install/build_and_install.md#build-and-install-hipvs-python-packages),
to see the list of dependencies and instructions on setting
up a conda environment for hipVS. Inside the environment, the examples can be run using
Python, for example:

```bash
$ python3 examples/python/cagra_example.py
```
