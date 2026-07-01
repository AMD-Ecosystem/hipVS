<!---
    MIT License

    Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
-->

# Changelog

hipVS is AMD's port of NVIDIA's cuVS (CUDA Vector Search) library, enabling approximate nearest neighbor (ANN) search algorithms on AMD GPUs using the HIP platform.

## [Version 2.0.0] - 2026-07-01

Upstream baseline upgraded from cuVS 25.10 to **cuVS 26.04** (~160 upstream commits), plus significant AMD-specific feature work and bug fixes. Requires **ROCm 7.2.4**.

### Features

**AMD-Specific**

- **Composable Kernel (CK) backend for fused distance NN:** New CK-accelerated fused nearest-neighbor kernel optimized for AMD GPUs, featuring skinny-K optimization, direct MFMA register reads in the epilogue, and proper LDS alignment. Supports fp32 data with L2, cosine, and inner-product metrics. Includes comprehensive test suite and IVF-realistic benchmarks
- **NN-Descent fp32 local_join_kernel ported to HIP:** Scalar fp32 `local_join_kernel` now runs natively on HIP (no rocWMMA dependency), restoring full NN-Descent fp32 recall (0.509 -> 0.969)
- **Rust (hipvs) bindings enabled:** The `hipvs` Rust crate is now built and tested as part of the standard release
- **CK/ck_tile ROCm 7.13 forward-compatibility:** Version-guarded support for the ROCm 7.13 `ck_tile` and `composable_kernel` API changes while maintaining backward compatibility with ROCm 7.2.x
- **ROCm 7.2.4 support:** Bumped base ROCm version from 7.2.1 to 7.2.4 across Dockerfiles and build infrastructure
- **IVF build profiling benchmark:** Google Benchmark-based IVF build profiling with optional rocprofiler-systems instrumentation

**Upstream (cuVS 26.04)**

- **PCA preprocessor:** Dimensionality reduction via PCA as a search preprocessor
- **KMeans batching:** Batch processing support for KMeans, plus inertia computation for balanced KMeans with weighted inertia option
- **IVF-PQ build factories:** C and Python APIs for IVF-PQ build with precomputed centroids and codebooks
- **CAGRA enhancements:** Float16 support for `cagra::extend`, L1 distance for iterative-search graph build, filtered `cagra::merge`, runtime template parameters to reduce binary size, iterative search as default graph build algorithm
- **HNSW:** GPU hierarchy used by default; ACE (Automatic Cross-links Elimination) build method with automatic partition count derivation
- **Spectral methods:** `all_neighbors` mode for spectral embedding, optional seed, improved `ncv` parameter edge-case handling, spectral clustering dataset API
- **DiskANN:** Configurable DRAM budget in SSD wrapper
- **NN-Descent:** fp16 data type support exposed to Python; overflow fix in `preprocess_data_kernel`
- **Rust bindings:** Vamana index bindings added upstream; `Index::search` changed to `&self`
- **Benchmark framework:** Pluggable backend architecture for cuvs-bench
- **C++ modernization:** Migrated to C++20, CCCL mdspan, `cuda::std::tuple`, `cuda::make_counting_iterator`; moved `faiss_select` from RAFT into cuVS

### Bug Fixes

**AMD-Specific**

- **CAGRA int8/uint8 multi-kernel Cosine:** Work-around for strict-aliasing issue in `args_t::load()` that caused incorrect results with the multi-kernel Cosine path on int8/uint8 data types
- **Fused distance NN stale-LDS fix:** Initialized `reduce_smem` to prevent stale LDS argmin keys from corrupting nearest-neighbor results
- **CK fused distance NN batch-local column index:** Fixed column index calculation that was using global rather than batch-local offsets
- **PQ sub-warp size:** Runtime sub-warp size selection for wave32/wave64 safety (fixes PQ with `pq_bits=5` on RDNA)
- **CAGRA bench shared memory:** Tightened smem constraint to fit within AMD's 64 KiB LDS limit
- **FAISS/bench-ann build:** Restored bench-ann build on HIP after the 26.04 upstream upgrade
- **C API fixes:** Fixed header check to use `project_root`, fixed bruteforce example type annotations, fixed C examples build on HIP

**Upstream**

- **Vamana serialization:** Fixed serialization bug
- **ScaNN:** Fixed AVQ prefetch
- **IVF-Flat:** Removed irrelevant assert in fused kernel mode
- **IVF-PQ:** Improved coarse batch size workspace estimation
- **Balanced KMeans:** Fixed `grid.y` overflow for >262K centroids by using `grid.x` in `adjust_centers`
- **CAGRA:** Fixed persistent regressions, graph degree bug, memory allocation in graph optimization, `cudaFuncSetAttribute` not called when switching kernel variants
- **DiskANN:** Fixed override handling
- **Binary quantizer:** Fixed host transform bounds and stream order
- **Integer overflow:** Fixed `get_free_host_memory()` on systems with >2 TB RAM

### Build & Infrastructure

- Upgraded upstream cuVS baseline from 25.10 to 26.04.00
- Bumped hipVS version to 2.0, amd-libhipvs dependency to 2.0
- ROCm base version bumped to 7.2.4
- Removed dead `CUDA_BACKEND` CMake scaffolding (AMD-only fork cleanup)
- Suppressed `-Wc2y-extensions` for Google Benchmark with HIP/clang
- Auto-set `HIPBLASLT_TENSILE_LIBPATH` for TheRock ROCm layout in CI
- C tests now run alongside C++ tests in CI
- Standardized AMD copyright headers across the project
- Updated documentation: v2.0.0 C/C++/Python/Rust API docs, Sphinx config, Doxygen generation

### Limitations

- Multi-Node / Multi-GPU is experimental
- wf32 (RDNA) is experimental

---

## [Version 1.0.0] - 2026-04-01

### Features
- Upgraded upstream hipVS baseline from 25.02 to 25.10, incorporating new upstream primitives, performance improvements, and API changes
- Added support for gfx950 AMD GPU architectures
- ROCm 7.2.1 support
- Enabled multi-GPU algorithms and RCCL configurations
- **Composable Kernel (CK) pairwise distance:** CK-accelerated L2, cosine, and inner-product distance kernels on AMD GPUs, supporting both C-order and F-order layouts including gfx950 float column-major paths
- **Benchmark visualization:** Static site generator for interactive benchmark result comparison
- **Large dataset support:** Batch processing for NN-Descent `local_join_kernel` to handle datasets exceeding the HIP runtime's 2^32 thread limit; conservative LDS-aware hash table sizing (`max_bitlen=11`) for AMD's 64 KB shared memory; CAGRA multi-CTA search fix (`itopk_size = 2*top_k`) to ensure sufficient CTAs for 64-wide wavefronts

### Bug Fixes

**Wavefront-width compatibility (wf32 and wf64):**
- IVF-Flat, IVF-PQ, NN-Descent, and Ball Cover: corrected lane-count assumptions for wf32 targets
- Ball Cover: resolved `eps_nn` hang and `eps_max_k` copy-kernel failures on 64-wide wavefronts
- NN-Descent: raised `max_iterations` ceiling and eliminated false early-termination on wf32

**Algorithm correctness:**
- Vamana: fixed segfault from incorrect edge counts and resolved doctest/pytest failures on HIP
- IVF-Flat: fixed uint8 cosine-distance recall regression; fixed MG `extend()` low recall with explicit indices
- NN Descent: fixed intermittent `ALL_NEIGHBORS_C_TEST` failure
- CK distance: fixed NaN self-distances in `l2_exp_ck_op` for F-order euclidean
- ScaNN: added `sync_stream` after buffer swap in quantization loop and `copy_stream` parameter to `apply_avq` to fix race conditions on HIP
- Spectral clustering: capped eigen_solver `max_iterations` to prevent NaN propagation into LAPACK's `steqr`
- CAGRA graph build: replaced `#pragma omp critical` with `std::mutex` in `merge_subgraphs` to fix OpenMP lock corruption across test iterations
- Restored missing `nnz_t` template parameter in `mutual_reachability_graph`

### Build & Infrastructure
- Update dependency branch names in versions.json
- Bump ROCm base version to 7.2.1 in build infrastructure

### Limitations
- Multi-Node/ Multi GPU is experimental
- wf32 is experimental

---

## [Initial Release Version 0.1.0] - 2025-11-04

### Platform Support
- Complete hipification of cuVS codebase for AMD GPUs
- ROCm 7.0 support with HIP platform
- AMD-specific optimizations (64-thread wavefront vs NVIDIA's 32)
- Integration with ROCmDS ecosystem
- Build system overhaul for HIP/ROCm with CMake and CPack
- Support for multiple AMD GPU architectures (gfx90a, gfx942)

### Algorithms
- **CAGRA:** Single/Multi-CTA search, graph building with IVF-PQ, support for `float`, `half`, `int8`, `uint8`
- **Vamana:** Graph building with robust pruning
- **IVF-Flat/IVF-PQ:** Complete implementation with dynamic batching
- **HNSW:** Full implementation
- **NN Descent:** Full implementation
- **Brute Force:** Complete baseline implementation
- **Distance Metrics:** L2, Inner Product, and others
- **Clustering:** K-means and spectral clustering

### Language Bindings
- **C:** Complete C API with test suite
- **C++:** Native templates and headers with examples
- **Python:** `hipvs` package with Cython bindings and examples
- **Rust:** `hipvs` crate (renamed from `cuvs`) with documentation

### Key Technical Changes
- Wavefront size adaptations (32->64 threads)
- Fixed uninitialized shared memory and bounds checking issues
- Optimized kernel launch parameters and LDS memory access for AMD GPUs
- Platform-specific intrinsic replacements (CUDA->HIP)
- Relaxed recall thresholds to account for hardware differences

### Multi-GPU Support (Experimental)
- Multi-GPU neighbors API with RCCL integration
- NEIGHBORS_MG_TEST suite enabled

### Contributors
- Lalith Narasimhan, Sujin Philip, Sukriti Choudhary, Grant Pinkert, Kevin Joseph, Randy Hartgrove, Alex Xu

---
