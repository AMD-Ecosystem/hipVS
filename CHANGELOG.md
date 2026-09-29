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

## [Version 2.0.0] - 2026-09-29

Upstream baseline upgraded from cuVS 25.10 to **cuVS 26.06**, with the cuVS 26.04 changes included, plus AMD-specific features, porting work, and fixes. Targets **ROCm 10.0.0**. The Composable Kernel (CK) paths stay compatible with ROCm 7.2.x, 7.13, and 7.14. Multi-node/multi-GPU and wave32 (RDNA) GPUs are now officially supported (both were experimental in 1.0.0).

### Features

**AMD-Specific**

- **Brute-force FP64:** End-to-end `float64` brute-force kNN (dense row- and column-major, sparse, `select_k`, `knn_merge_parts`, serialization) across the C++, C, and Python APIs. Fixes precision loss in the haversine and sparse Lp kernels. FP64 uses the tiled distance + `select_k` path because the fused L2 kNN kernel exceeds the 64 KiB LDS budget in double precision
- **JIT-LTO on ROCm:** cuVS 26.06's mandatory JIT-LTO subsystem is ported from nvJitLink/NVRTC to hipRTC/hipModule. This enables CAGRA, IVF-Flat, IVF-PQ, and IVF-SQ search and user-defined metrics (UDFs) on AMD GPUs. Bitcode is built per exact gfx architecture (portable multi-arch, concrete gfx90a), with faster bitcode embedding and an `-O3` link step
- **CK backend for fused distance NN:** CK-accelerated fused nearest-neighbor kernel (fp32; L2, cosine, inner product) with skinny-K optimization and MFMA-register epilogue
- **NN-Descent fp32 `local_join_kernel` on HIP:** Native scalar path, restoring fp32 recall (0.509 → 0.969)
- **Rust (`hipvs`) bindings:** Ported to the 26.06 API (including Vamana/DiskANN), built and tested as part of the release. The ROCm root is resolved from `ROCM_HOME`/`ROCM_PATH`
- **IVF build profiling benchmark:** Google Benchmark-based, with optional rocprofiler-systems instrumentation

**Upstream (cuVS 25.10 → 26.06)**

- **New indexes and APIs:** IVF-SQ (C++ and C APIs), UDF (user-defined distance) architecture, KDE kernel, PCA preprocessor with C and Python APIs, IVF-PQ build factories from precomputed centroids and codebooks
- **CAGRA:** Generalized `cagra::optimize`, JIT-LTO search, float16 `extend`, filtered `merge`, L1 for iterative graph build, iterative search as the default build algorithm
- **KMeans:** Batched and multi-GPU batched KMeans, weighted inertia for balanced KMeans, KMeans params exposed for PQ
- **NN-Descent:** L1 metric support, input memory location preserved, fp16 in Python
- **HNSW / all-neighbors:** GPU hierarchy by default, ACE build method, brute-force metrics exposed in `all_neighbors`
- **Rust:** CAGRA `search_with_filter` and serialize/deserialize, Vamana bindings
- **Other:** Spectral embedding `all_neighbors` mode, DiskANN DRAM budget, pluggable cuvs-bench backends, C++20 throughout, RMM migrated to the CCCL memory-resource design

### Bug Fixes

**AMD-Specific**

- **Sparse distances:** Stored zeros were double-counted by the union-based metrics (L1, L2Unexpanded, Linf, Canberra, Lp, Hamming, JensenShannon). Batched sparse kNN now pads short batches and keeps similarity ranking correct across the merge
- **Wave32 (RDNA) correctness, required for official wave32 support:** Warp-size fixes across CAGRA (incl. JIT kernels), IVF-PQ, IVF-Flat JIT, ScaNN, and `graph_core`. IVF-SQ scan skipped half of the data and serialize/search read out of bounds. CAGRA small-hash bitlen capped at 11 so the visited-hash table fits in 64 KiB LDS. PQ sub-warp size is now chosen at runtime
- **CK pairwise distance on RDNA3 (gfx11):** The CK path was never reached, and the wrong WMMA C layout scrambled output rows
- **hipBLAS/hipBLASLt limits:** Use `COMPUTE_32F` instead of TF32 (this unblocks IVF-PQ-based CAGRA builds). int8 GEMM goes through `COMPUTE_32I`. The int8 IVF-PQ coarse search is rejected with a clear error. The half alpha/beta workaround is extended to ROCm 10
- **NN-Descent:** WMMA multi-accumulator tiling for 8 warps (garbage Cosine/InnerProduct distances), degree taken from the device, convergence check debounced
- **JIT on multi-GPU / UDF:** Device ID added to the launcher cache key, filename-safe comgr names for UDF fragments, scalar byte ops in UDFs (no `__dp4a`/`__vabsdiff*`), `hipFuncGetAttribute` for module-loaded kernels
- **CAGRA:** Stream sync after dataset deserialize and strided-dataset copy, WAR for int8/uint8 multi-kernel Cosine, bench smem cap taken from the device's LDS limit
- **Other:** All-neighbors batch merge always uses SweepAll dedup. Zero-row `all_neighbors_graph` guarded (HNSW from ACE disk CAGRA). Stale-LDS argmin keys fixed in fused distance NN. CK fused NN batch-local column index fixed. More Lanczos search space for spectral clustering. Test fixes for the brute-force kNN reference NaN and KMeans `init_size` flakiness

**Upstream**

- **Security:** Hardened index deserialization against malicious inputs
- **CAGRA:** Concurrent dataset-descriptor initialization across streams, `optimize` mutating `raft::resources`, OOB access when graph size < dataset size, persistent-kernel and graph-degree regressions
- **Other:** argmin/argmax by distance type, PQ vectorized-load check, workspace usage, symbol exports (KMeans and others), brute-force Rust dataset lifetime, Vamana serialization, ScaNN AVQ prefetch, balanced KMeans `grid.y` overflow, binary quantizer bounds, `get_free_host_memory()` overflow on >2 TB hosts

### Build & Infrastructure

- Upstream baseline moved from cuVS 25.10 to 26.06, hipRAFT tracking `amd-integration`, hipVS and `amd-libhipvs` 2.0
- Dockerfiles moved to ROCm 10.0.0: the Ubuntu image uses the `rocm/dev-ubuntu-24.04:10.0.0-full` base, and the manylinux image installs `amdrocm-core-sdk10.0` from the new ROCm dnf repo (replacing `amdgpu-install`)
- New `TheRock.dockerfile`, with detection of TheRock's hipBLASLt `.dat.zlib` lazy index and auto-set `HIPBLASLT_TENSILE_LIBPATH`
- Python wheels built as a manylinux matrix (`cp311` + `cp312-abi3`, Python ≥ 3.11). `HIPVS_ENABLE_CK_BACKEND` toggle added
- CI scripts reorganized under `ci/internal/` (shellcheck-clean, streamed CTest output). C tests run alongside C++ tests
- Removed dead `CUDA_BACKEND` and CUDA-only JIT-LTO options. Python `VERSION` derived from the top-level `VERSION` file
- Docs rebranded to AMD Data Science and repo links moved to `AMD-Ecosystem/hipVS`. API reference gaps closed, Python stubs and Rust docs regenerated. Copyright headers standardized

### Limitations

- int8 IVF-PQ coarse search is unsupported (no hipBLAS i8×i8→f32 GEMM)
- FP64 brute force always uses the tiled (non-fused) kNN path

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
