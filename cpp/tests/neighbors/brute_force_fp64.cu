// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../../src/neighbors/detail/knn_brute_force.cuh"
#include "ann_utils.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::brute_force {
namespace {
using Metric = cuvs::distance::DistanceType;

// Independent host reference: never round input, arithmetic, or ranking keys to float.
double distance(const double* a, const double* b, int64_t dim, Metric metric)
{
  double squared = 0, dot = 0, aa = 0, bb = 0, l1 = 0, lp = 0;
  for (int64_t j = 0; j < dim; ++j) {
    const double diff = std::abs(a[j] - b[j]);
    squared += diff * diff;
    dot += a[j] * b[j];
    aa += a[j] * a[j];
    bb += b[j] * b[j];
    l1 += diff;
    lp += std::pow(diff, 3.0);
  }
  switch (metric) {
    case Metric::L2Expanded:
    case Metric::L2Unexpanded: return squared;
    case Metric::L2SqrtExpanded:
    case Metric::L2SqrtUnexpanded: return std::sqrt(squared);
    case Metric::InnerProduct: return dot;
    case Metric::CosineExpanded: return 1.0 - dot / std::sqrt(aa * bb);
    case Metric::L1: return l1;
    case Metric::LpUnexpanded: return std::pow(lp, 1.0 / 3.0);
    case Metric::Haversine: {
      const double lat = std::sin((a[0] - b[0]) * 0.5);
      const double lon = std::sin((a[1] - b[1]) * 0.5);
      return 2.0 * std::asin(std::sqrt(lat * lat + std::cos(a[0]) * std::cos(b[0]) * lon * lon));
    }
    default: throw std::logic_error("Missing FP64 test reference metric");
  }
}

class BruteForceFP64 : public ::testing::Test {
 protected:
  raft::resources res;

  template <typename T>
  auto upload(const std::vector<T>& values)
  {
    auto result = raft::make_device_vector<T, int64_t>(res, values.size());
    raft::update_device(
      result.data_handle(), values.data(), values.size(), raft::resource::get_cuda_stream(res));
    return result;
  }

  template <typename IdxT>
  void check(const std::vector<double>& dataset,
             const std::vector<double>& queries,
             int64_t dim,
             int64_t k,
             Metric metric,
             const IdxT* ids,
             const double* distances,
             const std::vector<bool>& allowed = {})
  {
    const int64_t n = dataset.size() / dim, m = queries.size() / dim;
    std::vector<IdxT> actual_ids(m * k);
    std::vector<double> actual_distances(m * k);
    auto stream = raft::resource::get_cuda_stream(res);
    raft::update_host(actual_ids.data(), ids, m * k, stream);
    raft::update_host(actual_distances.data(), distances, m * k, stream);
    raft::resource::sync_stream(res);
    for (int64_t q = 0; q < m; ++q) {
      std::vector<std::pair<double, IdxT>> expected;
      for (int64_t i = 0; i < n; ++i) {
        if (allowed.empty() || allowed[q * n + i]) {
          expected.emplace_back(distance(&queries[q * dim], &dataset[i * dim], dim, metric), i);
        }
      }
      std::sort(expected.begin(), expected.end(), [metric](auto a, auto b) {
        return metric == Metric::InnerProduct ? a.first > b.first : a.first < b.first;
      });
      ASSERT_GE(expected.size(), k);
      for (int64_t j = 0; j < k; ++j) {
        SCOPED_TRACE(::testing::Message() << "query=" << q << " rank=" << j);
        EXPECT_EQ(actual_ids[q * k + j], expected[j].second);
        EXPECT_NEAR(actual_distances[q * k + j],
                    expected[j].first,
                    2e-12 * std::max(1.0, std::abs(expected[j].first)));
      }
    }
  }

  std::vector<double> data(int64_t rows)
  {
    std::vector<double> result(rows * 2);
    for (int64_t i = 0; i < rows; ++i) {
      // All first coordinates round to the same float. Reversed order defeats
      // an implementation that breaks the resulting false ties by ascending ID.
      result[2 * i]     = 1.0 + (rows - i) * 1e-10;
      result[2 * i + 1] = 0.25;
    }
    return result;
  }

  std::vector<double> queries(int64_t rows)
  {
    std::vector<double> result(rows * 2);
    for (int64_t q = 0; q < rows; ++q) {
      result[2 * q]     = 0.125 + q * 0.001;
      result[2 * q + 1] = 0.75;
    }
    return result;
  }

  template <typename Layout>
  std::vector<double> layout(const std::vector<double>& values)
  {
    if constexpr (std::is_same_v<Layout, raft::row_major>) { return values; }
    std::vector<double> result(values.size());
    const auto rows = values.size() / 2;
    for (size_t i = 0; i < rows; ++i) {
      result[i]        = values[2 * i];
      result[rows + i] = values[2 * i + 1];
    }
    return result;
  }

  template <typename DataLayout, typename QueryLayout>
  void dense(Metric metric, int64_t k, bool host = false, int64_t n = 129)
  {
    constexpr int64_t m = 3, dim = 2;
    auto h_data = data(n), h_queries = queries(m);
    auto packed_data    = layout<DataLayout>(h_data);
    auto packed_queries = layout<QueryLayout>(h_queries);
    auto d_data = upload(packed_data), d_queries = upload(packed_queries);
    index_params params;
    params.metric     = metric;
    params.metric_arg = 3.0f;
    auto idx          = [&] {
      if (host) {
        return build(
          res, params, raft::make_host_matrix_view<const double, int64_t>(h_data.data(), n, dim));
      }
      return build(res,
                   params,
                   raft::make_device_matrix_view<const double, int64_t, DataLayout>(
                     d_data.data_handle(), n, dim));
    }();
    static_assert(std::is_same_v<decltype(idx), index<double, double>>);
    auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
    auto distances = raft::make_device_matrix<double, int64_t>(res, m, k);
    search(res,
           search_params{},
           idx,
           raft::make_device_matrix_view<const double, int64_t, QueryLayout>(
             d_queries.data_handle(), m, dim),
           ids.view(),
           distances.view());
    check(h_data, h_queries, dim, k, metric, ids.data_handle(), distances.data_handle());
    if (metric == Metric::L2Expanded || metric == Metric::CosineExpanded) {
      ASSERT_TRUE(idx.has_norms());
      std::vector<double> norms(n);
      raft::update_host(
        norms.data(), idx.norms().data_handle(), n, raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      double expected = h_data[0] * h_data[0] + h_data[1] * h_data[1];
      if (metric == Metric::CosineExpanded) { expected = std::sqrt(expected); }
      EXPECT_NEAR(norms[0], expected, 1e-14);
    }
  }
};

TEST_F(BruteForceFP64, DensePrecisionAndLayouts)
{
  for (auto metric : {Metric::L2Expanded,
                      Metric::L2Unexpanded,
                      Metric::L2SqrtExpanded,
                      Metric::L2SqrtUnexpanded,
                      Metric::InnerProduct,
                      Metric::CosineExpanded,
                      Metric::L1,
                      Metric::LpUnexpanded,
                      Metric::Haversine}) {
    for (int64_t k : {1, 31, 32, 33, 64, 65}) {
      SCOPED_TRACE(::testing::Message() << "metric=" << int(metric) << " k=" << k);
      dense<raft::row_major, raft::row_major>(metric, k);
      dense<raft::col_major, raft::row_major>(metric, k);
      dense<raft::row_major, raft::col_major>(metric, k);
      dense<raft::col_major, raft::col_major>(metric, k);
    }
    dense<raft::row_major, raft::row_major>(metric, 3, true);
  }
}

TEST_F(BruteForceFP64, MultipleColumnTiles)
{
  constexpr int64_t n = 129, m = 5, dim = 2, k = 7;
  auto h_data = data(n), h_queries = queries(m);
  auto d_data = upload(h_data), d_queries = upload(h_queries);
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
  auto distances = raft::make_device_matrix<double, int64_t>(res, m, k);
  // Force multiple row/column tiles without allocating a full default tile.
  // The last column tile has fewer than k entries.
  cuvs::neighbors::detail::tiled_brute_force_knn<double, int64_t, double>(res,
                                                                          d_queries.data_handle(),
                                                                          d_data.data_handle(),
                                                                          m,
                                                                          n,
                                                                          dim,
                                                                          k,
                                                                          distances.data_handle(),
                                                                          ids.data_handle(),
                                                                          Metric::L2Unexpanded,
                                                                          2.0,
                                                                          2,
                                                                          32);
  check(
    h_data, h_queries, dim, k, Metric::L2Unexpanded, ids.data_handle(), distances.data_handle());
}

TEST_F(BruteForceFP64, ManyQueriesCrossSelectKThreshold)
{
  // rows > 198 (with k > 2 and cols <= 22061) flips raft's select_k heuristic from
  // kWarpImmediate to kWarpDistributedShm (see choose_select_k_algorithm). Pairing that
  // with a forced column tile split exercises both the select_k row threshold and the
  // column-tiling cutover for double in the same call.
  constexpr int64_t n = 129, m = 199, dim = 2, k = 7;
  auto h_data = data(n), h_queries = queries(m);
  auto d_data = upload(h_data), d_queries = upload(h_queries);
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
  auto distances = raft::make_device_matrix<double, int64_t>(res, m, k);
  cuvs::neighbors::detail::tiled_brute_force_knn<double, int64_t, double>(res,
                                                                          d_queries.data_handle(),
                                                                          d_data.data_handle(),
                                                                          m,
                                                                          n,
                                                                          dim,
                                                                          k,
                                                                          distances.data_handle(),
                                                                          ids.data_handle(),
                                                                          Metric::L2Unexpanded,
                                                                          2.0,
                                                                          0,
                                                                          32);
  check(
    h_data, h_queries, dim, k, Metric::L2Unexpanded, ids.data_handle(), distances.data_handle());
}

TEST_F(BruteForceFP64, MultiPartitionMerge)
{
  // The public build()/search() API always calls brute_force_knn_impl with a single dataset
  // partition, so knn_merge_parts (which combines per-partition top-k results) never runs for
  // double through that path. Call brute_force_knn_impl directly with two partitions to
  // exercise it.
  constexpr int64_t n = 129, n1 = 70, m = 5, dim = 2, k = 7;
  auto h_data = data(n), h_queries = queries(m);
  auto d_data = upload(h_data), d_queries = upload(h_queries);
  std::vector<double*> input = {d_data.data_handle(), d_data.data_handle() + n1 * dim};
  std::vector<int64_t> sizes = {n1, n - n1};
  auto ids                   = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
  auto distances             = raft::make_device_matrix<double, int64_t>(res, m, k);
  cuvs::neighbors::detail::brute_force_knn_impl<int64_t, int64_t, double, double>(
    res,
    input,
    sizes,
    dim,
    d_queries.data_handle(),
    m,
    ids.data_handle(),
    distances.data_handle(),
    k,
    true,
    true,
    nullptr,
    Metric::L2Unexpanded,
    2.0);
  check(
    h_data, h_queries, dim, k, Metric::L2Unexpanded, ids.data_handle(), distances.data_handle());
}

TEST_F(BruteForceFP64, FusedPathNeverSelectedForDouble)
{
  // hipRaft's Policy2x8<double, 32, *> contraction policy is documented as unused ("just for
  // keeping compiler happy"), so fusedL2KnnFitsSharedMem must reject double regardless of the
  // device's warp size, forcing every double brute-force search through the tiled path.
  auto stream = raft::resource::get_cuda_stream(res);
  for (int k : {1, 32, 64}) {
    EXPECT_FALSE(
      (cuvs::neighbors::detail::fusedL2KnnFitsSharedMem<double, double, int64_t>(k, stream)));
  }
}

TEST_F(BruteForceFP64, FilteredPrecision)
{
  constexpr int64_t n = 256, dim = 2, k = 3;
  for (auto metric :
       {Metric::L2Expanded, Metric::L2SqrtExpanded, Metric::InnerProduct, Metric::CosineExpanded}) {
    for (int64_t m : {3, 12}) {
      for (bool bitmap : {false, true}) {
        for (int stride : {2, 20}) {  // 50% and 95% sparsity
          SCOPED_TRACE(::testing::Message() << "metric=" << int(metric) << " m=" << m
                                            << " bitmap=" << bitmap << " stride=" << stride);
          auto h_data = data(n), h_queries = queries(m);
          auto d_data = upload(h_data), d_queries = upload(h_queries);
          index_params params;
          params.metric = metric;
          auto idx      = build(
            res,
            params,
            raft::make_device_matrix_view<const double, int64_t>(d_data.data_handle(), n, dim));
          std::vector<uint32_t> words((bitmap ? m * n : n) / 32, 0);
          std::vector<bool> allowed(m * n);
          for (int64_t q = 0; q < m; ++q) {
            for (int64_t i = 0; i < n; ++i) {
              if ((i + (bitmap ? q : 0)) % stride != 0) { continue; }
              allowed[q * n + i] = true;
              auto bit           = bitmap ? q * n + i : i;
              words[bit / 32] |= uint32_t{1} << (bit % 32);
            }
          }
          auto mask      = upload(words);
          auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
          auto distances = raft::make_device_matrix<double, int64_t>(res, m, k);
          auto query_view =
            raft::make_device_matrix_view<const double, int64_t>(d_queries.data_handle(), m, dim);
          if (bitmap) {
            filtering::bitmap_filter<uint32_t, int64_t> filter(
              cuvs::core::bitmap_view<uint32_t, int64_t>(mask.data_handle(), m, n));
            search(res, search_params{}, idx, query_view, ids.view(), distances.view(), filter);
          } else {
            filtering::bitset_filter<uint32_t, int64_t> filter(
              cuvs::core::bitset_view<uint32_t, int64_t>(mask.data_handle(), n));
            search(res, search_params{}, idx, query_view, ids.view(), distances.view(), filter);
          }
          check(
            h_data, h_queries, dim, k, metric, ids.data_handle(), distances.data_handle(), allowed);
        }
      }
    }
  }
}

TEST_F(BruteForceFP64, SparsePrecisionAndBatches)
{
  constexpr int n = 19, m = 5, dim = 2, k = 3;
  auto h_data = data(n), h_queries = queries(m);
  // Include an empty CSR row in each input.
  h_data[0] = h_data[1] = h_queries[0] = h_queries[1] = 0;
  std::vector<int> data_ptr(n + 1), query_ptr(m + 1), data_cols, query_cols;
  for (int i = 1; i < n; ++i) {
    data_cols.insert(data_cols.end(), {0, 1});
    data_ptr[i + 1] = data_cols.size();
  }
  for (int i = 1; i < m; ++i) {
    query_cols.insert(query_cols.end(), {0, 1});
    query_ptr[i + 1] = query_cols.size();
  }
  auto dp = upload(data_ptr), qp = upload(query_ptr), dc = upload(data_cols),
       qc      = upload(query_cols);
  auto dv      = upload(std::vector<double>(h_data.begin() + 2, h_data.end()));
  auto qv      = upload(std::vector<double>(h_queries.begin() + 2, h_queries.end()));
  auto dataset = raft::make_device_csr_matrix_view<const double>(
    dv.data_handle(),
    raft::make_device_compressed_structure_view<int, int, int>(
      dp.data_handle(), dc.data_handle(), n, dim, int(data_cols.size())));
  auto query = raft::make_device_csr_matrix_view<const double>(
    qv.data_handle(),
    raft::make_device_compressed_structure_view<int, int, int>(
      qp.data_handle(), qc.data_handle(), m, dim, int(query_cols.size())));
  // L2/L1 have unique distances even for the empty query row.
  for (auto metric : {Metric::L2Expanded,
                      Metric::L2Unexpanded,
                      Metric::L2SqrtExpanded,
                      Metric::L2SqrtUnexpanded,
                      Metric::L1,
                      Metric::LpUnexpanded}) {
    auto idx = build(res, dataset, metric, 3.0f);
    static_assert(std::is_same_v<decltype(idx), sparse_index<double, int>>);
    for (int batch : {6, 7, 64}) {
      sparse_search_params params;
      params.batch_size_index = batch;
      params.batch_size_query = batch < 64 ? 2 : 64;
      auto ids                = raft::make_device_matrix<int, int64_t>(res, m, k);
      auto distances          = raft::make_device_matrix<double, int64_t>(res, m, k);
      search(res, params, idx, query, ids.view(), distances.view());
      check(h_data, h_queries, dim, k, metric, ids.data_handle(), distances.data_handle());
    }
  }
}

TEST_F(BruteForceFP64, SparseSimilarityMetrics)
{
  constexpr int n = 19, m = 5, dim = 2, k = 3;
  auto h_data = data(n), h_queries = queries(m);
  std::vector<int> data_ptr(n + 1), query_ptr(m + 1), data_cols(n * dim), query_cols(m * dim);
  for (int i = 0; i <= n; ++i) {
    data_ptr[i] = i * dim;
  }
  for (int i = 0; i <= m; ++i) {
    query_ptr[i] = i * dim;
  }
  for (int i = 0; i < n * dim; ++i) {
    data_cols[i] = i % dim;
  }
  for (int i = 0; i < m * dim; ++i) {
    query_cols[i] = i % dim;
  }
  auto dp = upload(data_ptr), qp = upload(query_ptr), dc = upload(data_cols),
       qc = upload(query_cols);
  auto dv = upload(h_data), qv = upload(h_queries);
  auto dataset = raft::make_device_csr_matrix_view<const double>(
    dv.data_handle(),
    raft::make_device_compressed_structure_view<int, int, int>(
      dp.data_handle(), dc.data_handle(), n, dim, n * dim));
  auto query = raft::make_device_csr_matrix_view<const double>(
    qv.data_handle(),
    raft::make_device_compressed_structure_view<int, int, int>(
      qp.data_handle(), qc.data_handle(), m, dim, m * dim));
  for (auto metric : {Metric::InnerProduct, Metric::CosineExpanded}) {
    auto idx = build(res, dataset, metric);
    for (int batch : {6, 64}) {
      sparse_search_params params;
      params.batch_size_index = batch;
      params.batch_size_query = 2;
      auto ids                = raft::make_device_matrix<int, int64_t>(res, m, k);
      auto distances          = raft::make_device_matrix<double, int64_t>(res, m, k);
      search(res, params, idx, query, ids.view(), distances.view());
      check(h_data, h_queries, dim, k, metric, ids.data_handle(), distances.data_handle());
    }
  }
}

TEST_F(BruteForceFP64, HaversineLargeK)
{
  dense<raft::row_major, raft::row_major>(Metric::Haversine, 1024, false, 1025);
}

TEST_F(BruteForceFP64, Serialization)
{
  constexpr int64_t n = 17, m = 3, dim = 2, k = 3;
  auto h_data = data(n), h_queries = queries(m);
  auto d_queries = upload(h_queries);
  for (auto metric : {Metric::L2Expanded, Metric::InnerProduct}) {
    index_params params;
    params.metric = metric;
    auto idx =
      build(res, params, raft::make_host_matrix_view<const double, int64_t>(h_data.data(), n, dim));
    for (bool include_dataset : {false, true}) {
      for (bool file : {false, true}) {
        index<double, double> restored(res);
        if (file) {
          cuvs::neighbors::tmp_index_file tmp;
          serialize(res, tmp.filename, idx, include_dataset);
          deserialize(res, tmp.filename, &restored);
        } else {
          std::stringstream buffer;
          serialize(res, buffer, idx, include_dataset);
          deserialize(res, buffer, &restored);
          buffer.seekg(0);
          index<float, float> wrong_type(res);
          EXPECT_THROW(deserialize(res, buffer, &wrong_type), raft::logic_error);
        }
        EXPECT_EQ(restored.metric(), metric);
        EXPECT_EQ(restored.metric_arg(), double(params.metric_arg));
        EXPECT_EQ(restored.has_norms(), idx.has_norms());
        if (!include_dataset) {
          restored.update_dataset(
            res, raft::make_host_matrix_view<const double, int64_t>(h_data.data(), n, dim));
        }
        EXPECT_EQ(restored.size(), n);
        auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
        auto distances = raft::make_device_matrix<double, int64_t>(res, m, k);
        search(
          res,
          search_params{},
          restored,
          raft::make_device_matrix_view<const double, int64_t>(d_queries.data_handle(), m, dim),
          ids.view(),
          distances.view());
        check(h_data, h_queries, dim, k, metric, ids.data_handle(), distances.data_handle());
      }
    }
  }
}
}  // namespace
}  // namespace cuvs::neighbors::brute_force
