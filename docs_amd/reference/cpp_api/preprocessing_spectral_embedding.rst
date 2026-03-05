..
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

Spectral Embedding
==================

Spectral embedding is a powerful dimensionality reduction technique that uses the eigenvectors
of the graph Laplacian to embed high-dimensional data into a lower-dimensional space. This
method is particularly effective for discovering non-linear manifold structures in data and
is widely used in clustering, visualization, and feature extraction tasks.

.. role:: py(code)
   :language: c++
   :class: highlight

Overview
--------

The spectral embedding algorithm works by:

1. **Graph Construction**: Building a k-nearest neighbors graph from the input data
2. **Laplacian Computation**: Computing the graph Laplacian matrix (normalized or unnormalized)
3. **Eigendecomposition**: Finding the eigenvectors corresponding to the smallest eigenvalues
4. **Embedding**: Using these eigenvectors as coordinates in the lower-dimensional space

Parameters
----------

``#include <cuvs/preprocessing/spectral_embedding.hpp>``

namespace *cuvs::preprocessing::spectral_embedding*

.. doxygenstruct:: cuvs::preprocessing::spectral_embedding::params
   :project: cuvs
   :members:

Functions
---------

``#include <cuvs/preprocessing/spectral_embedding.hpp>``

namespace *cuvs::preprocessing::spectral_embedding*

.. doxygengroup:: spectral_embedding
   :project: cuvs
   :content-only:

Example Usage
-------------

Basic Usage with Dataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <raft/core/resources.hpp>
   #include <cuvs/preprocessing/spectral_embedding.hpp>

   // Initialize RAFT resources
   raft::resources handle;

   // Configure spectral embedding parameters
   cuvs::preprocessing::spectral_embedding::params params;
   params.n_components = 2;        // Reduce to 2D for visualization
   params.n_neighbors = 15;        // Local neighborhood size
   params.norm_laplacian = true;   // Use normalized Laplacian
   params.drop_first = true;       // Drop constant eigenvector
   params.seed = 42;               // For reproducibility

   // Create input dataset (n_samples x n_features)
   int n_samples = 1000;
   int n_features = 50;
   auto dataset = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
   // ... populate dataset with your data ...

   // Allocate output embedding matrix (n_samples x n_components)
   auto embedding = raft::make_device_matrix<float, int, raft::col_major>(
       handle, n_samples, params.n_components);

   // Perform spectral embedding
   cuvs::preprocessing::spectral_embedding::transform(
       handle, params, dataset.view(), embedding.view());

Using Precomputed Graph
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <raft/core/resources.hpp>
   #include <cuvs/preprocessing/spectral_embedding.hpp>

   raft::resources handle;

   // Configure parameters (n_neighbors is ignored with precomputed graph)
   cuvs::preprocessing::spectral_embedding::params params;
   params.n_components = 3;
   params.norm_laplacian = true;
   params.drop_first = true;
   params.seed = 42;

   // Assume we have a precomputed connectivity graph
   // This could be from custom similarity computation or k-NN search
   raft::device_coo_matrix<float, int, int, int> connectivity_graph(...);

   // Allocate output embedding
   auto embedding = raft::make_device_matrix<float, int, raft::col_major>(
       handle, n_samples, params.n_components);

   // Perform spectral embedding with precomputed graph
   cuvs::preprocessing::spectral_embedding::transform(
       handle, params, connectivity_graph.view(), embedding.view());
