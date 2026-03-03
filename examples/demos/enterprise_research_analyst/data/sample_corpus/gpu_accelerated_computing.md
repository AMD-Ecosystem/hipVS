# GPU-Accelerated Computing

Graphics Processing Units (GPUs) were originally designed for rendering pixels, but their massively parallel architecture makes them ideal for general-purpose computing. A modern data-center GPU contains thousands of compute cores, high-bandwidth memory (HBM), and hardware schedulers that can execute millions of lightweight threads concurrently.

## SIMT Execution Model

GPUs use a Single-Instruction, Multiple-Thread (SIMT) model where groups of threads (called wavefronts on AMD or warps on NVIDIA) execute the same instruction in lockstep. This is perfect for data-parallel workloads like matrix multiplication, distance computation, and embedding generation.

## Memory Hierarchy

HBM provides bandwidth exceeding 3 TB/s on the latest AMD MI300X accelerators, enabling rapid streaming of large vector datasets. On-chip shared memory (LDS on AMD) allows cooperative data reuse within a workgroup, which is critical for tiled matrix operations used in vector search kernels.

## Applications Beyond Graphics

Today GPUs power deep-learning training and inference, scientific simulation, financial modelling, genomics, and -- increasingly -- vector-database workloads where brute-force or ANN search must run at interactive latencies over massive datasets.
