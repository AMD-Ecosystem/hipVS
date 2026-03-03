# AMD Instinct Accelerators

AMD Instinct is a family of data-center GPUs designed for AI training, inference, and high-performance computing. The latest generation, MI300X, combines the CDNA 3 compute architecture with 192 GB of HBM3 memory and over 5 TB/s of memory bandwidth.

## CDNA Architecture

Unlike AMD's consumer RDNA GPUs, the CDNA architecture is optimised for matrix and vector compute. It features dedicated Matrix Fused Multiply-Add (MFMA) units that accelerate GEMM operations central to deep learning and distance computation.

## ROCm Software Stack

ROCm is AMD's open-source GPU-compute platform. It provides HIP (a portable C++ runtime), math libraries (rocBLAS, hipBLAS), and deep-learning framework support (PyTorch, TensorFlow). hipVS and hipRAFT are built on ROCm, bringing GPU-accelerated vector search to AMD hardware.

## Multi-GPU Scaling

AMD Instinct GPUs support high-bandwidth Infinity Fabric links for multi-GPU communication. hipVS can distribute vector indexes across multiple GPUs (MG-CAGRA, MG-IVF-Flat, MG-IVF-PQ) to handle billion-scale datasets that exceed single-GPU memory.
