/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::ffi::CString;
use std::io::{Write, stderr};

use crate::dlpack::ManagedTensor;
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;
use crate::vamana::IndexParams;

/// Vamana ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsVamanaIndex_t);

impl Index {
    /// Builds Vamana Index for efficient DiskANN search
    ///
    /// The build uses the Vamana insertion-based algorithm to create the graph. The algorithm
    /// starts with an empty graph and iteratively inserts batches of nodes. Each batch involves
    /// performing a greedy search for each vector to be inserted, and inserting it with edges to
    /// all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied
    /// to improve graph quality. The index_params struct controls the degree of the final graph.
    ///
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build<T: Into<ManagedTensor>>(
        res: &Resources,
        params: &IndexParams,
        dataset: T,
    ) -> Result<Index> {
        let dataset: ManagedTensor = dataset.into();
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsVamanaBuild(res.0, params.0, dataset.as_ptr(), index.0))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsVamanaIndex_t>::uninit();
            check_cuvs(ffi::cuvsVamanaIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Save Vamana index to file
    ///
    /// Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.
    ///
    /// Serialized Index is to be used by the DiskANN open-source repository for graph search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file prefix for where the index is sazved
    /// * `include_dataset` - whether to include the dataset in the serialized index
    pub fn serialize(self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()> {
        let c_filename = CString::new(filename).unwrap();
        unsafe {
            check_cuvs(ffi::cuvsVamanaSerialize(
                res.0,
                c_filename.as_ptr(),
                self.0,
                include_dataset,
            ))
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsVamanaIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsVamanaIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_vamana() {
        // graph_degree must be one of the supported sizes {32, 64, 128, 256} and be >= the
        // device warp/wavefront size (64 on CDNA), and visited_size must be > graph_degree.
        // The upstream defaults (graph_degree=32, visited_size=64) are only valid on a 32-wide
        // warp, so set wavefront-safe values here.
        let build_params = IndexParams::new()
            .unwrap()
            .set_graph_degree(64)
            .set_visited_size(128);

        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        // build the vamana index
        let _index = Index::build(&res, &build_params, dataset_device)
            .expect("failed to create vamana index");
    }
}
