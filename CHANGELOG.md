# Changelog

hipVS is AMD's port of NVIDIA's cuVS (CUDA Vector Search) library, enabling approximate nearest neighbor (ANN) search algorithms on AMD GPUs using the HIP platform.

## [Version 1.0.0] - 2026-04-01

### Features
- Upgraded upstream hipVS baseline from 25.02 to 25.10, incorporating new upstream primitives, performance improvements, and API changes
- Added support for gfx950 AMD GPU architectures
- ROCm 7.2.3 support
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
- Bump ROCm base version to 7.2.3 in build infrastructure

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
- Wavefront size adaptations (32→64 threads)
- Fixed uninitialized shared memory and bounds checking issues
- Optimized kernel launch parameters and LDS memory access for AMD GPUs
- Platform-specific intrinsic replacements (CUDA→HIP)
- Relaxed recall thresholds to account for hardware differences

### Multi-GPU Support (Experimental)
- Multi-GPU neighbors API with RCCL integration
- NEIGHBORS_MG_TEST suite enabled

### Contributors
- Lalith Narasimhan, Sujin Philip, Sukriti Choudhary, Grant Pinkert, Kevin Joseph, Randy Hartgrove, Alex Xu
