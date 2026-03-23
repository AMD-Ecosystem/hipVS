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

What is hipVS?
==============

hipVS is an API providing GPU-accelerated vector search functions. Vector search is an information
retrieval method that has been growing in use over the past few years, partly because of the
rising importance of multimedia embeddings created from unstructured data and the need to perform
semantic search on the embeddings to find items which are semantically similar to each other. 
Vector search is also used in data mining and machine learning tasks. 

It comprises an important step in many clustering and visualization algorithms like `UMAP <https://arxiv.org/abs/2008.00325>`_,
`t-SNE <https://lvdmaaten.github.io/tsne/>`_, K-means, and `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html>`_.
Faster vector search enables interactions between dense vectors and graphs. Converting a pile
of dense vectors into nearest neighbors graphs unlocks graph analysis algorithms,
such as those found in `GraphBLAS <https://graphblas.org/>`_.

Below are some common use-cases for vector search:

* Semantic search

  - Generative AI & Retrieval-Augmented Generation (RAG)
  - Recommender systems
  - Computer vision
  - Image search
  - Text search
  - Audio search
  - Molecular search
  - Model training

* Data mining

  - Clustering algorithms
  - Visualization algorithms
  - Sampling algorithms
  - Class balancing
  - Ensemble methods
  - k-NN graph construction

Benefits of hipVS
-----------------

Following are some benefits of using hipVS for AMD GPU-accelerated vector search:

1. Fast index build
2. Latency critical and high throughput search
3. Parameter tuning
4. Cost savings
5. Interoperability (build on GPU, deploy on CPU)
6. Multiple language support
7. Building blocks for composing new or accelerating existing algorithms

Highlights
----------

* Approximate & exact nearest-neighbor search – HNSW, IVF-PQ, brute force and more
* C++, C, Rust, and Python interfaces – integrate in low-latency services or rapid prototyping notebooks
* API-compatible with cuVS – drop-in replacement for existing cuVS workflows on AMD GPUs

Limitations
-----------

* Multi-GPU and Multinode functionality is experimental.

References
----------

Many of the accelerated implementations in hipVS are also based on research papers which provide additional background. You are encouraged to cite the corresponding algorithms by referencing them in your own research.

- `CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search <https://arxiv.org/abs/2308.15136>`_
- `Top-K Algorithms on GPU: A Comprehensive Study and New Methods <https://dl.acm.org/doi/10.1145/3581784.3607062>`_
- `Fast K-NN Graph Construction by GPU Based NN-Descent <https://dl.acm.org/doi/abs/10.1145/3459637.3482344?casa_token=O_nan1B1F5cAAAAA:QHWDEhh0wmd6UUTLY9_Gv6c3XI-5DXM9mXVaUXOYeStlpxTPmV3nKvABRfoivZAaQ3n8FWyrkWw>`_
- `cuSLINK: Single-linkage Agglomerative Clustering on the GPU <https://arxiv.org/abs/2306.16354>`_
- `GPU Semiring Primitives for Sparse Neighborhood Methods <https://arxiv.org/abs/2104.06357>`_
