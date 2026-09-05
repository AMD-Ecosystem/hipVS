/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include "../test_utils.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/stats/silhouette_score.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>

namespace cuvs {
namespace stats {

// parameter structure definition
struct silhouetteScoreParam {
  int nRows;
  int nCols;
  int nLabels;
  cuvs::distance::DistanceType metric;
  int chunk;
  double tolerance;
};

// test fixture class
template <typename LabelT, typename DataT>
class silhouetteScoreTest : public ::testing::TestWithParam<silhouetteScoreParam> {
 protected:
  silhouetteScoreTest()
    : d_X(0, raft::resource::get_cuda_stream(handle)),
      sampleSilScore(0, raft::resource::get_cuda_stream(handle)),
      d_labels(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void host_silhouette_score()
  {
    // Generating random value test input
    std::vector<double> h_X(nElements);  // Each element will be default initialized. That is 0.0.
    std::vector<int> h_labels(nRows);    // Each element will be default initialized. That is 0.
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(0, nLabels - 1);
    std::uniform_real_distribution<double> realGenerator(0.0, 100.0);

    std::generate(h_X.begin(), h_X.end(), [&]() { return realGenerator(dre); });
    std::generate(h_labels.begin(), h_labels.end(), [&]() { return intGenerator(dre); });

    // allocating and initializing memory to the GPU
    auto stream = raft::resource::get_cuda_stream(handle);
    d_X.resize(nElements, stream);
    d_labels.resize(nRows, stream);
    // HIP/AMD: We don't need to zero initialize by calling cudaMemsetAsync here as we're populating
    // dX and d_labels with host data coming from h_X and h_labels.
    sampleSilScore.resize(nElements, stream);

    raft::update_device(d_X.data(), h_X.data(), nElements, stream);
    raft::update_device(d_labels.data(),
                        h_labels.data(),
                        nRows,
                        stream);  // HIP/AMD: This was incorrectly using nElements upstream.

    // finding the distance matrix

    rmm::device_uvector<double> d_distanceMatrix(nRows * nRows, stream);

    auto d_X_view = raft::make_device_matrix_view<const DataT, int64_t>(d_X.data(), nRows, nCols);
    cuvs::distance::pairwise_distance(
      handle,
      d_X_view,
      d_X_view,
      raft::make_device_matrix_view<DataT, int64_t>(d_distanceMatrix.data(), nRows, nRows),
      params.metric);

    raft::resource::sync_stream(handle, stream);

    std::vector<double> h_distanceMatrix(nRows * nRows);
    raft::update_host(
      h_distanceMatrix.data(), d_distanceMatrix.data(), h_distanceMatrix.size(), stream);

    // finding the bincount array
    std::vector<std::size_t> binCountArray(nLabels, 0);
    for (int lbl : h_labels)
      binCountArray[lbl]++;

    std::vector<double> a(nRows);
    std::vector<double> b(nRows);
    static constexpr double MAX = std::numeric_limits<double>::max();

    for (int i = 0; i < nRows; ++i) {
      int myLabel               = h_labels[i];
      double sumOfIntraClusterD = 0;

      for (int j = 0; j < nRows; ++j) {
        if (h_labels[j] == myLabel) { sumOfIntraClusterD += h_distanceMatrix[i * nRows + j]; }
      }

      if (binCountArray[myLabel] <= 1)
        a[i] = -1;
      else
        a[i] = sumOfIntraClusterD / (binCountArray[myLabel] - 1);
    }

    // finding the average inter cluster distance for every element

    for (int i = 0; i < nRows; ++i) {
      int myLabel          = h_labels[i];
      double minAvgInterCD = MAX;

      for (int j = 0; j < nLabels; ++j) {
        int curClLabel = j;
        if (curClLabel == myLabel) continue;
        double avgInterCD = 0;

        for (int k = 0; k < nRows; ++k) {
          if (h_labels[k] == curClLabel) { avgInterCD += h_distanceMatrix[i * nRows + k]; }
        }

        if (binCountArray[curClLabel])
          avgInterCD /= binCountArray[curClLabel];
        else
          avgInterCD = MAX;
        minAvgInterCD = std::min(minAvgInterCD, avgInterCD);
      }

      b[i] = minAvgInterCD;
    }

    // finding the silhouette score for every element

    std::vector<double> truthSampleSilScore(nRows);
    for (int i = 0; i < nRows; ++i) {
      if (a[i] == -1)
        truthSampleSilScore[i] = 0;
      else if (a[i] == 0 && b[i] == 0)
        truthSampleSilScore[i] = 0;
      else
        truthSampleSilScore[i] = (b[i] - a[i]) / std::max(a[i], b[i]);
      truthSilhouetteScore += truthSampleSilScore[i];
    }

    truthSilhouetteScore /= nRows;
  }

  // the constructor
  void SetUp() override
  {
    // getting the parameters
    params = ::testing::TestWithParam<silhouetteScoreParam>::GetParam();

    nRows     = params.nRows;
    nCols     = params.nCols;
    nLabels   = params.nLabels;
    chunk     = params.chunk;
    nElements = nRows * nCols;

    host_silhouette_score();

    // calling the silhouette_score CUDA implementation
    computedSilhouetteScore = cuvs::stats::silhouette_score(
      handle,
      raft::make_device_matrix_view<const DataT>(d_X.data(), nRows, nCols),
      raft::make_device_vector_view<const LabelT>(d_labels.data(), nRows),
      std::make_optional(raft::make_device_vector_view(sampleSilScore.data(), nRows)),
      nLabels,
      params.metric);

    batchedSilhouetteScore = cuvs::stats::silhouette_score_batched(
      handle,
      raft::make_device_matrix_view<const DataT>(d_X.data(), nRows, nCols),
      raft::make_device_vector_view<const LabelT>(d_labels.data(), nRows),
      std::make_optional(raft::make_device_vector_view(sampleSilScore.data(), nRows)),
      nLabels,
      chunk,
      params.metric);
  }

  // declaring the data values
  raft::resources handle;
  silhouetteScoreParam params;
  int nLabels;
  rmm::device_uvector<DataT> d_X;
  rmm::device_uvector<DataT> sampleSilScore;
  rmm::device_uvector<LabelT> d_labels;
  int nRows;
  int nCols;
  int nElements;
  double truthSilhouetteScore    = 0;
  double computedSilhouetteScore = 0;
  double batchedSilhouetteScore  = 0;
  int chunk;
};

// setting test parameter values
const std::vector<silhouetteScoreParam> inputs = {
  {4, 2, 3, cuvs::distance::DistanceType::L2Expanded, 4, 0.00001},
  {4, 2, 2, cuvs::distance::DistanceType::L2SqrtUnexpanded, 2, 0.00001},
  {8, 8, 3, cuvs::distance::DistanceType::L2Unexpanded, 4, 0.00001},
  {11, 2, 5, cuvs::distance::DistanceType::L2Expanded, 3, 0.00001},
  {40, 2, 8, cuvs::distance::DistanceType::L2Expanded, 10, 0.00001},
  {12, 7, 3, cuvs::distance::DistanceType::CosineExpanded, 8, 0.00001},
  {7, 5, 5, cuvs::distance::DistanceType::L1, 2, 0.00001},
  // HIP/AMD: The three cases below guard compute_chunked_a_b's and
  // fill_b_kernel's block-size clamping (silhouette_score.cuh) against the
  // 64-wide AMD wavefront. Both kernels used to size a block as
  // min(extent, warp_size) per dimension, which is safe at warp_size == 32
  // (32 x 32 == 1024 threads, exactly maxThreadsPerBlock) but launches an
  // illegal >1024-thread block at warp_size == 64 whenever both extents
  // reach the wavefront width. On NVIDIA these are ordinary correctness
  // checks; on AMD (warp_size 64) they fail with hipErrorInvalidConfiguration
  // without the clamp.
  // dist_rows == dist_cols == chunk == 70 exceeds a 64-wide wavefront in
  // compute_chunked_a_b while nLabels stays small enough to keep
  // fill_b_kernel's block legal either way.
  {70, 3, 3, cuvs::distance::DistanceType::L2Expanded, 70, 0.00001},
  // nLabels == 20 exceeds fill_b_kernel's legal block on a 64-wide wavefront
  // while chunk == 32 keeps compute_chunked_a_b's block legal either way.
  {96, 4, 20, cuvs::distance::DistanceType::L2Expanded, 32, 0.00001},
  // nRows == chunk == 80 and nLabels == 24 push both kernels over the limit
  // on a 64-wide wavefront at the same time.
  {80, 3, 24, cuvs::distance::DistanceType::L2Expanded, 80, 0.00001}};

// writing the test suite
typedef silhouetteScoreTest<int, double> silhouetteScoreTestClass;
TEST_P(silhouetteScoreTestClass, Result)
{
  ASSERT_NEAR(computedSilhouetteScore, truthSilhouetteScore, params.tolerance);
  ASSERT_NEAR(batchedSilhouetteScore, truthSilhouetteScore, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(silhouetteScore, silhouetteScoreTestClass, ::testing::ValuesIn(inputs));

}  // end namespace stats
}  // end namespace cuvs
