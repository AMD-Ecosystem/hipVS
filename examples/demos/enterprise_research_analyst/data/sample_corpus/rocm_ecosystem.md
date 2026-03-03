# The ROCm Software Ecosystem

ROCm (Radeon Open Compute) is AMD's open-source software platform for GPU computing. It provides compilers, runtime libraries, math libraries, and framework integrations that enable developers to run CUDA-like workloads on AMD GPUs.

## HIP: Portable GPU Programming

HIP (Heterogeneous-compute Interface for Portability) is a C++ runtime API that closely mirrors CUDA. Code written in HIP can target both AMD and NVIDIA GPUs, simplifying porting. hipVS uses HIP to bring RAPIDS cuVS functionality to AMD hardware.

## hipRAFT and hipVS

hipRAFT is a port of NVIDIA's RAFT (Reusable Accelerated Functions and Tools) library. It provides GPU-accelerated primitives for distance computation, clustering, and linear algebra. hipVS builds on hipRAFT to deliver high-performance vector search algorithms (CAGRA, IVF-Flat, IVF-PQ) on AMD Instinct GPUs.

## Framework Support

ROCm supports PyTorch, TensorFlow, and JAX, enabling end-to-end AI pipelines on AMD hardware. Combined with hipVS for vector search, this allows building complete RAG applications entirely on AMD GPUs.
