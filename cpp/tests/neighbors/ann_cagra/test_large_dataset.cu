// MIT License
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../ann_cagra.cuh"
#include "../ann_utils.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuvs/neighbors/cagra.hpp>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

static std::tuple<raft::device_uvector<uint32_t>, raft::device_uvector<float>> compute_groundtruth(
  raft::device_uvector<float> const& dataset,
  raft::device_uvector<float> const& queries,
  size_t rows_dataset,
  size_t rows_queries,
  size_t dims,
  size_t top_k,
  cuvs::distance::DistanceType metric)
{
  auto handle  = raft::resources{};
  auto stream  = raft::resource::get_cuda_stream(handle);
  const auto k = std::min(top_k, rows_dataset);

  raft::device_uvector<uint32_t> neighbors_dev(rows_queries * k, stream);
  raft::device_uvector<float> distances_dev(rows_queries * k, stream);

  cuvs::neighbors::naive_knn<float, float, uint32_t>(handle,
                                                     distances_dev.data(),
                                                     neighbors_dev.data(),
                                                     queries.data(),
                                                     dataset.data(),
                                                     rows_queries,
                                                     rows_dataset,
                                                     dims,
                                                     static_cast<uint32_t>(k),
                                                     metric);

  raft::resource::sync_stream(handle);
  return std::make_tuple(std::move(neighbors_dev), std::move(distances_dev));
}

struct TestParameter {
  int top_k;
  int dim;
  cuvs::distance::DistanceType metric;
  size_t dataset_size;
  size_t query_size;
  float min_recall;
};

class TestBenchMarkDatasets : public ::testing::TestWithParam<TestParameter> {
 public:
  TestBenchMarkDatasets()
    : m_handle(),
      m_dataset_dev(0, raft::resource::get_cuda_stream(m_handle)),
      m_queries_dev(0, raft::resource::get_cuda_stream(m_handle)),
      m_neighbors_dev(0, raft::resource::get_cuda_stream(m_handle)),
      m_distances_dev(0, raft::resource::get_cuda_stream(m_handle)),
      m_gt_neighbors(0, raft::resource::get_cuda_stream(m_handle)),
      m_gt_distances(0, raft::resource::get_cuda_stream(m_handle))
  {
  }

 protected:
  void SetUp() override
  {
    m_dataset_dev.resize(GetParam().dataset_size * GetParam().dim);
    m_queries_dev.resize(GetParam().query_size * GetParam().dim);
    m_neighbors_dev.resize(GetParam().query_size * GetParam().top_k);
    m_distances_dev.resize(GetParam().query_size * GetParam().top_k);
    raft::random::RngState r(1234ULL);
    cuvs::neighbors::cagra::GenerateRoundingErrorFreeDataset(
      m_handle, m_dataset_dev.data(), GetParam().dataset_size, GetParam().dim, r, false);
    cuvs::neighbors::cagra::GenerateRoundingErrorFreeDataset(
      m_handle, m_queries_dev.data(), GetParam().query_size, GetParam().dim, r, false);
    raft::resource::sync_stream(m_handle);
    std::tie(m_gt_neighbors, m_gt_distances) = compute_groundtruth(m_dataset_dev,
                                                                   m_queries_dev,
                                                                   GetParam().dataset_size,
                                                                   GetParam().query_size,
                                                                   GetParam().dim,
                                                                   GetParam().top_k,
                                                                   GetParam().metric);
  }

  raft::resources m_handle;
  raft::device_uvector<uint32_t> m_gt_neighbors;
  raft::device_uvector<float> m_gt_distances;
  raft::device_uvector<float> m_dataset_dev;
  raft::device_uvector<float> m_queries_dev;
  raft::device_uvector<uint32_t> m_neighbors_dev;
  raft::device_uvector<float> m_distances_dev;
};

auto get_nn_descent_params(int graph_degree, cuvs::distance::DistanceType metric)
  -> cuvs::neighbors::cagra::graph_build_params::nn_descent_params
{
  auto params         = cuvs::neighbors::cagra::graph_build_params::nn_descent_params();
  params.graph_degree = graph_degree * 2;
  params.intermediate_graph_degree = params.graph_degree * 2;
  params.max_iterations            = 100;
  params.return_distances          = true;
  params.metric                    = metric;
  return params;
}

TEST_P(TestBenchMarkDatasets, TestLargeDataset)
{
  cuvs::neighbors::cagra::index<float, uint32_t> index{m_handle, GetParam().metric};
  auto build_params   = cuvs::neighbors::cagra::index_params();
  build_params.metric = GetParam().metric;
  build_params.graph_build_params =
    get_nn_descent_params(build_params.graph_degree, GetParam().metric);
  auto dataset_view_device = raft::make_device_matrix_view<const float, uint32_t>(
    m_dataset_dev.data(), GetParam().dataset_size, GetParam().dim);
  index = cuvs::neighbors::cagra::build(m_handle, build_params, dataset_view_device);
  // Intentionally wait for the build to complete. Throw an error if it fails.
  raft::resource::sync_stream(m_handle);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto queries_view = raft::make_device_matrix_view<const float, int64_t>(
    m_queries_dev.data(), GetParam().query_size, GetParam().dim);
  auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
    m_neighbors_dev.data(), GetParam().query_size, GetParam().top_k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(
    m_distances_dev.data(), GetParam().query_size, GetParam().top_k);
  auto search_params = cuvs::neighbors::cagra::search_params();
  search_params.itopk_size =
    std::max(search_params.itopk_size, static_cast<size_t>(GetParam().top_k * 2));
  cuvs::neighbors::cagra::search(m_handle,
                                 search_params,
                                 index,
                                 queries_view,
                                 neighbors_view,
                                 distances_view,
                                 cuvs::neighbors::filtering::none_sample_filter{});
  std::vector<uint32_t> neighbors_actual(GetParam().query_size * GetParam().top_k);
  raft::copy(neighbors_actual.data(),
             m_neighbors_dev.data(),
             neighbors_actual.size(),
             raft::resource::get_cuda_stream(m_handle));
  std::vector<float> distances_actual(GetParam().query_size * GetParam().top_k);
  raft::copy(distances_actual.data(),
             m_distances_dev.data(),
             distances_actual.size(),
             raft::resource::get_cuda_stream(m_handle));
  std::vector<uint32_t> gt_neighbors(GetParam().query_size * GetParam().top_k);
  raft::copy(gt_neighbors.data(),
             m_gt_neighbors.data(),
             gt_neighbors.size(),
             raft::resource::get_cuda_stream(m_handle));
  std::vector<float> gt_distances(GetParam().query_size * GetParam().top_k);
  raft::copy(gt_distances.data(),
             m_gt_distances.data(),
             gt_distances.size(),
             raft::resource::get_cuda_stream(m_handle));
  raft::resource::sync_stream(m_handle);
  EXPECT_TRUE(cuvs::neighbors::eval_recall(gt_neighbors,
                                           neighbors_actual,
                                           GetParam().query_size,
                                           GetParam().top_k,
                                           0.01,
                                           GetParam().min_recall,
                                           false));
}

INSTANTIATE_TEST_SUITE_P(TestBenchMarkDatasets,
                         TestBenchMarkDatasets,
                         ::testing::Values(
                           TestParameter{
                             .top_k        = 64,
                             .dim          = 96,
                             .metric       = cuvs::distance::DistanceType::L2Expanded,
                             .dataset_size = 4194304,
                             .query_size   = 100,
                             .min_recall   = 0.1,
                           },
                           TestParameter{
                             .top_k        = 16,
                             .dim          = 96,
                             .metric       = cuvs::distance::DistanceType::L2Expanded,
                             .dataset_size = 10000,
                             .query_size   = 100,
                             .min_recall   = 0.9,
                           },
                           TestParameter{
                             .top_k        = 16,
                             .dim          = 96,
                             .metric       = cuvs::distance::DistanceType::L2Expanded,
                             .dataset_size = 4194303,
                             .query_size   = 100,
                             .min_recall   = 0.1,
                           }));
