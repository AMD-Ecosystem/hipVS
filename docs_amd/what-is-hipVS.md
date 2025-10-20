# What is hipVS?

hipVS is an API providing GPU-accelerated vector search functions. Vector search is an information
retrieval method that has been growing in use over the past few years, partly because of the
rising importance of multimedia embeddings created from unstructured data and the need to perform
semantic search on the embeddings to find items which are semantically similar to each other. 
Vector search is also used in data mining and machine learning tasks. 

It comprises an important step in many clustering and visualization algorithms like [UMAP](https://arxiv.org/abs/2008.00325),
[t-SNE](https://lvdmaaten.github.io/tsne/), K-means, and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).
Faster vector search enables interactions between dense vectors and graphs. Converting a pile
of dense vectors into nearest neighbors graphs unlocks graph analysis algorithms,
such as those found in [GraphBLAS](https://graphblas.org/).

Below are some common use-cases for vector search:

* Semantic search - 

  - Generative AI & Retrieval-Augmented Generation (RAG)
  - Recommender systems
  - Computer vision
  - Image search
  - Text search
  - Audio search
  - Molecular search
  - Model training

* Data mining - 

  - Clustering algorithms
  - Visualization algorithms
  - Sampling algorithms
  - Class balancing
  - Ensemble methods
  - k-NN graph construction

## Benefits of hipVS 

Following are some benefits of using hipVS for AMD GPU-accelerated vector search:

1. Fast index build
2. Latency critical and high throughput search
3. Parameter tuning
4. Cost savings
5. Interoperability (build on GPU, deploy on CPU)
6. Multiple language support
7. Building blocks for composing new or accelerating existing algorithms

##  Highlights

* Approximate & exact nearest-neighbor search – HNSW, IVF-PQ, brute force and more
* C++, C, Rust, and Python interfaces – integrate in low-latency services or rapid prototyping notebooks
* API-compatible with cuVS – drop-in replacement for existing cuVS workflows on AMD GPUs

##  Limitations

* Multi-GPU and Multinode functionality is experimental.

## References

Many of the accelerated implementations in hipVS are also based on research papers which provide additional background. You are encouraged to cite the corresponding algorithms by referencing them in your own research.
- [CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search](https://arxiv.org/abs/2308.15136)
- [Top-K Algorithms on GPU: A Comprehensive Study and New Methods](https://dl.acm.org/doi/10.1145/3581784.3607062)
- [Fast K-NN Graph Construction by GPU Based NN-Descent](https://dl.acm.org/doi/abs/10.1145/3459637.3482344?casa_token=O_nan1B1F5cAAAAA:QHWDEhh0wmd6UUTLY9_Gv6c3XI-5DXM9mXVaUXOYeStlpxTPmV3nKvABRfoivZAaQ3n8FWyrkWw>)
- [cuSLINK: Single-linkage Agglomerative Clustering on the GPU](https://arxiv.org/abs/2306.16354)
- [GPU Semiring Primitives for Sparse Neighborhood Methods](https://arxiv.org/abs/2104.06357)
