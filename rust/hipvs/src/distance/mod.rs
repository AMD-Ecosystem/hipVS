/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

use crate::distance_type::DistanceType;
use crate::dlpack::ManagedTensor;
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;

/// Compute pairwise distances between X and Y
///
/// # Arguments
///
/// * `res` - Resources to use
/// * `x` - A matrix in device memory - shape (m, k)
/// * `y` - A matrix in device memory - shape (n, k)
/// * `distances` - A matrix in device memory that receives the output distances - shape (m, n)
/// * `metric` - DistanceType to use for building the index
/// * `metric_arg` - Optional value of `p` for Minkowski distances
pub fn pairwise_distance(
    res: &Resources,
    x: &ManagedTensor,
    y: &ManagedTensor,
    distances: &ManagedTensor,
    metric: DistanceType,
    metric_arg: Option<f32>,
) -> Result<()> {
    unsafe {
        check_cuvs(ffi::cuvsPairwiseDistance(
            res.0,
            x.as_ptr(),
            y.as_ptr(),
            distances.as_ptr(),
            metric,
            metric_arg.unwrap_or(2.0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_pairwise_distance() {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_datapoints, n_datapoints));
        let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        pairwise_distance(
            &res,
            &dataset_device,
            &dataset_device,
            &distances,
            DistanceType::L2Expanded,
            None,
        )
        .unwrap();

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();

        // Self distance should be 0
        assert_eq!(distances_host[[0, 0]], 0.0);
    }
}
