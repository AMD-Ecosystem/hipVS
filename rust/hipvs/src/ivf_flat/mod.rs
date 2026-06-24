/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//! The IVF-Flat method is an ANN algorithm. It uses an inverted file index (IVF) with
//! unmodified (that is, flat) vectors. This algorithm provides simple knobs to reduce
//! the overall search space and to trade-off accuracy for speed.
//!
//! Example:
//! ```
//!
//! use hipvs::ivf_flat::{Index, IndexParams, SearchParams};
//! use hipvs::{ManagedTensor, Resources, Result};
//!
//! use ndarray::s;
//! use ndarray_rand::rand_distr::Uniform;
//! use ndarray_rand::RandomExt;
//!
//! fn ivf_flat_example() -> Result<()> {
//!     let res = Resources::new()?;
//!
//!     // Create a new random dataset to index
//!     let n_datapoints = 65536;
//!     let n_features = 512;
//!     let dataset =
//!         ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
//!
//!     // build the ivf-flat index
//!     let build_params = IndexParams::new()?;
//!     let index = Index::build(&res, &build_params, &dataset)?;
//!     println!(
//!         "Indexed {}x{} datapoints into ivf-flat index",
//!         n_datapoints, n_features
//!     );
//!
//!     // use the first 4 points from the dataset as queries : will test that we get them back
//!     // as their own nearest neighbor
//!     let n_queries = 4;
//!     let queries = dataset.slice(s![0..n_queries, ..]);
//!
//!     let k = 10;
//!
//!     // Ivf-Flat search API requires queries and outputs to be on device memory
//!     // copy query data over, and allocate new device memory for the distances/ neighbors
//!     // outputs
//!     let queries = ManagedTensor::from(&queries).to_device(&res)?;
//!     let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
//!     let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;
//!
//!     let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
//!     let distances = ManagedTensor::from(&distances_host).to_device(&res)?;
//!
//!     let search_params = SearchParams::new()?;
//!
//!     index.search(&res, &search_params, &queries, &neighbors, &distances)?;
//!
//!     // Copy back to host memory
//!     distances.to_host(&res, &mut distances_host)?;
//!     neighbors.to_host(&res, &mut neighbors_host)?;
//!
//!     // nearest neighbors should be themselves, since queries are from the
//!     // dataset
//!     println!("Neighbors {:?}", neighbors_host);
//!     println!("Distances {:?}", distances_host);
//!     Ok(())
//! }
//! ```

mod index;
mod index_params;
mod search_params;

pub use index::Index;
pub use index_params::IndexParams;
pub use search_params::SearchParams;
