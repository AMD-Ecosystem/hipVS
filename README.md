# hipVS

**GPU-accelerated vector search for AMD GPUs**

## What is hipVS?

hipVS is a port of the RAPIDS **[cuvs](https://github.com/rapidsai/cuvs)** library. It aims to
follow the latter's directory structure, file naming and API naming as closely as possible to
minimize porting friction for users that are interested in using both projects.

It contains state-of-the-art implementations of several algorithms for running approximate nearest neighbors and clustering on the GPU. It can be used directly or through the various databases and other libraries that have integrated it. The primary goal of hipVS is to simplify the use of GPUs for vector similarity search and clustering.

Vector search is an information retrieval method that has been growing in popularity over the past few  years, partly because of the rising importance of multimedia embeddings created from unstructured data and the need to perform semantic search on the embeddings to find items which are semantically similar to each other.

Vector search is also used in _data mining and machine learning_ tasks and comprises an important step in many _clustering_ and _visualization_ algorithms like [UMAP](https://arxiv.org/abs/2008.00325), [t-SNE](https://lvdmaaten.github.io/tsne/), K-means, and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).

Finally, faster vector search enables interactions between dense vectors and graphs. Converting a pile of dense vectors into nearest neighbors graphs unlocks the entire world of graph analysis algorithms, such as those found in [GraphBLAS](https://graphblas.org/).

Below are some common use-cases for vector search:

- ### Semantic search
  - Generative AI & Retrieval augmented generation (RAG)
  - Recommender systems
  - Computer vision
  - Image search
  - Text search
  - Audio search
  - Molecular search
  - Model training


- ### Data mining
  - Clustering algorithms
  - Visualization algorithms
  - Sampling algorithms
  - Class balancing
  - Ensemble methods
  - k-NN graph construction

## Why hipVS?

There are several benefits to using hipVS and GPUs for vector search, including

1. Fast index build
2. Latency critical and high throughput search
3. Parameter tuning
4. Cost savings
5. Interoperability (build on GPU, deploy on CPU)
6. Multiple language support
7. Building blocks for composing new or accelerating existing algorithms

## Getting Started

The following code snippets train an approximate nearest neighbors index for the CAGRA algorithm in the various different languages supported by hipVS.

### Python API

```python
from cuvs.neighbors import cagra

dataset = load_data()
index_params = cagra.IndexParams()

index = cagra.build(build_params, dataset)
```

##  Highlights

* **Approximate & exact nearest-neighbor search** – HNSW, IVF-PQ, brute force and more
* **C++, C & Python interfaces** – integrate in low-latency services or rapid prototyping notebooks
* **API-compatible with cuVS** – drop-in replacement for existing cuVS workflows on AMD hardware

##  Limitations

* The library is currently only tested on AMD GPUs with the ROCm stack.
* Multi-GPU and Multi Node functionality is still experimental and not officially supported.

### C++ API

```cpp
cagra::index_params index_params;

auto index = cagra::build(res, index_params, dataset);
```

### C API

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraIndexParams_t index_params;
cuvsCagraIndex_t index;

DLManagedTensor *dataset;
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsCagraIndexParamsCreate(&index_params);
cuvsCagraIndexCreate(&index);

cuvsCagraBuild(res, index_params, dataset, index);

cuvsCagraIndexDestroy(index);
cuvsCagraIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

### Rust API

```rust
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources, Result};

use ndarray::s;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Example showing how to index and search data with CAGRA
fn cagra_example() -> Result<()> {
    let res = Resources::new()?;

    // Create a new random dataset to index
    let n_datapoints = 65536;
    let n_features = 512;
    let dataset =
        ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

    // build the cagra index
    let build_params = IndexParams::new()?;
    let index = Index::build(&res, &build_params, &dataset)?;
    println!(
        "Indexed {}x{} datapoints into cagra index",
        n_datapoints, n_features
    );

    // use the first 4 points from the dataset as queries : will test that we get them back
    // as their own nearest neighbor
    let n_queries = 4;
    let queries = dataset.slice(s![0..n_queries, ..]);

    let k = 10;

    // CAGRA search API requires queries and outputs to be on device memory
    // copy query data over, and allocate new device memory for the distances/ neighbors
    // outputs
    let queries = ManagedTensor::from(&queries).to_device(&res)?;
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(&res)?;

    let search_params = SearchParams::new()?;

    index.search(&res, &search_params, &queries, &neighbors, &distances)?;

    // Copy back to host memory
    distances.to_host(&res, &mut distances_host)?;
    neighbors.to_host(&res, &mut neighbors_host)?;

    // nearest neighbors should be themselves, since queries are from the
    // dataset
    println!("Neighbors {:?}", neighbors_host);
    println!("Distances {:?}", distances_host);
    Ok(())
}
```

## References

For the interested reader, many of the accelerated implementations in hipVS are also based on research papers which can provide a lot more background. We also ask you to please cite the corresponding algorithms by referencing them in your own research.
- [CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search](https://arxiv.org/abs/2308.15136)
- [Top-K Algorithms on GPU: A Comprehensive Study and New Methods](https://dl.acm.org/doi/10.1145/3581784.3607062)
- [Fast K-NN Graph Construction by GPU Based NN-Descent](https://dl.acm.org/doi/abs/10.1145/3459637.3482344?casa_token=O_nan1B1F5cAAAAA:QHWDEhh0wmd6UUTLY9_Gv6c3XI-5DXM9mXVaUXOYeStlpxTPmV3nKvABRfoivZAaQ3n8FWyrkWw>)
- [cuSLINK: Single-linkage Agglomerative Clustering on the GPU](https://arxiv.org/abs/2306.16354)
- [GPU Semiring Primitives for Sparse Neighborhood Methods](https://arxiv.org/abs/2104.06357)
- [VecFlow: A High-Performance Vector Data Management System for Filtered-Search on GPUs](https://arxiv.org/abs/2506.00812)

---
