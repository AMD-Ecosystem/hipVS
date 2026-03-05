# Changelog

hipVS is AMD's port of NVIDIA's cuVS (CUDA Vector Search) library, enabling approximate nearest neighbor (ANN) search algorithms on AMD GPUs using the HIP platform.

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
