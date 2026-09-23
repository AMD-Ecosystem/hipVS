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
#include <raft/util/integer_utils.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// This file exercises the FP64 (double) brute-force path. It is organized top-to-bottom as:
//   1. Shared test infrastructure  - the host reference oracle and the fixture base helpers.
//   2. Public-API suites           - dense, pre-filtered, sparse, and serialization searches,
//                                    each a value-parametrized fixture.
//   3. Implementation-detail paths - the handful of tests that call detail:: internals directly
//                                    to reach code the public API never routes double through.

namespace cuvs::neighbors::brute_force {
namespace {
using Metric = cuvs::distance::DistanceType;

// ============================================================================
// Shared test infrastructure
// ============================================================================

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

// Bare-identifier metric name for parameterized test IDs and failure output. print_metric from
// ann_utils.cuh emits "distance::L2Expanded", whose "::" is illegal in a gtest test name, so this
// returns the plain symbol instead.
std::string metric_label(Metric metric)
{
  switch (metric) {
    case Metric::L2Expanded: return "L2Expanded";
    case Metric::L2Unexpanded: return "L2Unexpanded";
    case Metric::L2SqrtExpanded: return "L2SqrtExpanded";
    case Metric::L2SqrtUnexpanded: return "L2SqrtUnexpanded";
    case Metric::InnerProduct: return "InnerProduct";
    case Metric::CosineExpanded: return "CosineExpanded";
    case Metric::L1: return "L1";
    case Metric::LpUnexpanded: return "LpUnexpanded";
    case Metric::Haversine: return "Haversine";
    default: return "Metric" + std::to_string(int(metric));
  }
}

// Shared helpers for the FP64 brute-force tests. Kept off ::testing::Test so it can be
// mixed into both the plain (TEST_F) fixture and the value-parametrized (TEST_P) fixture
// below without a diamond over the gtest base.
class BruteForceFP64Base {
 protected:
  raft::resources res;

  // Every dataset/query buffer this fixture builds is a fixed, row-major matrix with kDim
  // columns; kDim is therefore the per-row stride into the flattened std::vector<double>
  // buffers produced by data()/queries(). The Haversine reference additionally treats
  // column 0 as latitude and column 1 as longitude, so kDim must stay 2.
  static constexpr int64_t kDim = 2;

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
    std::vector<double> result(rows * kDim);
    for (int64_t i = 0; i < rows; ++i) {
      // All first coordinates round to the same float. Reversed order defeats
      // an implementation that breaks the resulting false ties by ascending ID.
      result[kDim * i]     = 1.0 + (rows - i) * 1e-10;
      result[kDim * i + 1] = 0.25;
    }
    return result;
  }

  std::vector<double> queries(int64_t rows)
  {
    std::vector<double> result(rows * kDim);
    for (int64_t q = 0; q < rows; ++q) {
      result[kDim * q]     = 0.125 + q * 0.001;
      result[kDim * q + 1] = 0.75;
    }
    return result;
  }

  std::vector<double> pack_as(const std::vector<double>& values, bool col_major)
  {
    if (!col_major) { return values; }
    std::vector<double> result(values.size());
    const auto rows = values.size() / kDim;
    // Row-major -> column-major transpose of a (rows x kDim) matrix.
    for (size_t i = 0; i < rows; ++i) {
      for (int64_t c = 0; c < kDim; ++c) {
        result[c * rows + i] = values[i * kDim + c];
      }
    }
    return result;
  }
};

// ============================================================================
// Dense brute force (public API)
// ============================================================================

// One dense-search configuration. Layouts are runtime flags rather than template
// parameters because build()/search() expose separate row_major/col_major overloads
// (see build_index()/the search dispatch below), not a single layout-templated entry.
struct DenseParam {
  Metric metric;
  int64_t k;
  int64_t n;
  bool host;
  bool data_col_major;
  bool query_col_major;
};

std::ostream& operator<<(std::ostream& os, const DenseParam& p)
{
  return os << "{metric=" << metric_label(p.metric) << " k=" << p.k << " n=" << p.n
            << " host=" << p.host << " data_col_major=" << p.data_col_major
            << " query_col_major=" << p.query_col_major << '}';
}

class DenseFP64 : public BruteForceFP64Base, public ::testing::TestWithParam<DenseParam> {
 protected:
  static constexpr int64_t m                     = 3;
  DenseParam p                                   = GetParam();
  std::vector<double> h_data                     = data(p.n);
  std::vector<double> h_queries                  = queries(m);
  raft::device_vector<double, int64_t> d_data    = upload(pack_as(h_data, p.data_col_major));
  raft::device_vector<double, int64_t> d_queries = upload(pack_as(h_queries, p.query_col_major));
  index<double, double> idx                      = build_index();

  // Built once per parameter combination in the fixture's member-init list. The host
  // build ignores the (row-major) data layout, matching the original coverage.
  index<double, double> build_index()
  {
    index_params params;
    params.metric     = p.metric;
    params.metric_arg = 3.0f;
    if (p.host) {
      return build(
        res, params, raft::make_host_matrix_view<const double, int64_t>(h_data.data(), p.n, kDim));
    }
    if (p.data_col_major) {
      return build(res,
                   params,
                   raft::make_device_matrix_view<const double, int64_t, raft::col_major>(
                     d_data.data_handle(), p.n, kDim));
    }
    return build(res,
                 params,
                 raft::make_device_matrix_view<const double, int64_t, raft::row_major>(
                   d_data.data_handle(), p.n, kDim));
  }
};

TEST_P(DenseFP64, Search)
{
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, p.k);
  auto distances = raft::make_device_matrix<double, int64_t>(res, m, p.k);
  if (p.query_col_major) {
    search(res,
           search_params{},
           idx,
           raft::make_device_matrix_view<const double, int64_t, raft::col_major>(
             d_queries.data_handle(), m, kDim),
           ids.view(),
           distances.view());
  } else {
    search(res,
           search_params{},
           idx,
           raft::make_device_matrix_view<const double, int64_t, raft::row_major>(
             d_queries.data_handle(), m, kDim),
           ids.view(),
           distances.view());
  }
  check(h_data, h_queries, kDim, p.k, p.metric, ids.data_handle(), distances.data_handle());
  if (p.metric == Metric::L2Expanded || p.metric == Metric::CosineExpanded) {
    ASSERT_TRUE(idx.has_norms());
    std::vector<double> norms(p.n);
    raft::update_host(
      norms.data(), idx.norms().data_handle(), p.n, raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    double expected = h_data[0] * h_data[0] + h_data[1] * h_data[1];
    if (p.metric == Metric::CosineExpanded) { expected = std::sqrt(expected); }
    EXPECT_NEAR(norms[0], expected, 1e-14);
  }
}

std::vector<DenseParam> dense_params()
{
  std::vector<DenseParam> v;
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
      for (bool data_col_major : {false, true}) {
        for (bool query_col_major : {false, true}) {
          v.push_back({metric, k, 129, false, data_col_major, query_col_major});
        }
      }
    }
    // One host build per metric (row-major, k=3), matching the original coverage.
    v.push_back({metric, 3, 129, true, false, false});
  }
  // Large-k Haversine case (formerly the HaversineLargeK test): n must exceed k.
  v.push_back({Metric::Haversine, 1024, 1025, false, false, false});
  return v;
}

std::string dense_name(const ::testing::TestParamInfo<DenseParam>& info)
{
  const auto& p = info.param;
  return metric_label(p.metric) + "_k" + std::to_string(p.k) + "_n" + std::to_string(p.n) +
         (p.host ? "_host" : "_dev") + (p.data_col_major ? "_dcol" : "_drow") +
         (p.query_col_major ? "_qcol" : "_qrow");
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseFP64, ::testing::ValuesIn(dense_params()), dense_name);

// ============================================================================
// Pre-filtered search (public API)
// ============================================================================

// One filtered-search configuration shared by the bitset and bitmap tests below.
// stride sets the sparsity (every stride-th index is allowed).
struct FilterParam {
  Metric metric;
  int64_t m;
  int stride;
};

std::ostream& operator<<(std::ostream& os, const FilterParam& p)
{
  return os << "{metric=" << metric_label(p.metric) << " m=" << p.m << " stride=" << p.stride
            << '}';
}

class FilteredFP64 : public BruteForceFP64Base, public ::testing::TestWithParam<FilterParam> {
 protected:
  static constexpr int64_t n = 256, k = 3;
  FilterParam p                                  = GetParam();
  std::vector<double> h_data                     = data(n);
  std::vector<double> h_queries                  = queries(p.m);
  raft::device_vector<double, int64_t> d_data    = upload(h_data);
  raft::device_vector<double, int64_t> d_queries = upload(h_queries);
  index<double, double> idx                      = build_index();

  index<double, double> build_index()
  {
    index_params params;
    params.metric = p.metric;
    return build(
      res,
      params,
      raft::make_device_matrix_view<const double, int64_t>(d_data.data_handle(), n, kDim));
  }

  // Search with the given filter and verify against the host reference restricted to
  // the allowed (query, index) pairs.
  void run_and_check(const filtering::base_filter& filter, const std::vector<bool>& allowed)
  {
    auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, p.m, k);
    auto distances = raft::make_device_matrix<double, int64_t>(res, p.m, k);
    auto query_view =
      raft::make_device_matrix_view<const double, int64_t>(d_queries.data_handle(), p.m, kDim);
    search(res, search_params{}, idx, query_view, ids.view(), distances.view(), filter);
    check(
      h_data, h_queries, kDim, k, p.metric, ids.data_handle(), distances.data_handle(), allowed);
  }
};

// A single bitset shared by every query: index i is allowed when i % stride == 0.
TEST_P(FilteredFP64, Bitset)
{
  std::vector<uint32_t> words(raft::div_rounding_up_safe<int64_t>(n, 32), 0);
  std::vector<bool> allowed(p.m * n);
  for (int64_t i = 0; i < n; ++i) {
    if (i % p.stride != 0) { continue; }
    words[i / 32] |= uint32_t{1} << (i % 32);
    for (int64_t q = 0; q < p.m; ++q) {
      allowed[q * n + i] = true;
    }
  }
  auto mask = upload(words);
  filtering::bitset_filter<uint32_t, int64_t> filter(
    cuvs::core::bitset_view<uint32_t, int64_t>(mask.data_handle(), n));
  run_and_check(filter, allowed);
}

// A per-query bitmap: (query q, index i) is allowed when (i + q) % stride == 0.
TEST_P(FilteredFP64, Bitmap)
{
  std::vector<uint32_t> words(raft::div_rounding_up_safe<int64_t>(p.m * n, 32), 0);
  std::vector<bool> allowed(p.m * n);
  for (int64_t q = 0; q < p.m; ++q) {
    for (int64_t i = 0; i < n; ++i) {
      if ((i + q) % p.stride != 0) { continue; }
      auto bit     = q * n + i;
      allowed[bit] = true;
      words[bit / 32] |= uint32_t{1} << (bit % 32);
    }
  }
  auto mask = upload(words);
  filtering::bitmap_filter<uint32_t, int64_t> filter(
    cuvs::core::bitmap_view<uint32_t, int64_t>(mask.data_handle(), p.m, n));
  run_and_check(filter, allowed);
}

std::vector<FilterParam> filter_params()
{
  std::vector<FilterParam> v;
  for (auto metric :
       {Metric::L2Expanded, Metric::L2SqrtExpanded, Metric::InnerProduct, Metric::CosineExpanded}) {
    for (int64_t m : {3, 12}) {
      for (int stride : {2, 20}) {  // 50% and 95% sparsity
        v.push_back({metric, m, stride});
      }
    }
  }
  return v;
}

std::string filter_name(const ::testing::TestParamInfo<FilterParam>& info)
{
  const auto& p = info.param;
  return metric_label(p.metric) + "_q" + std::to_string(p.m) + "_stride" + std::to_string(p.stride);
}

INSTANTIATE_TEST_SUITE_P(Filtered, FilteredFP64, ::testing::ValuesIn(filter_params()), filter_name);

// ============================================================================
// Sparse brute force (public API)
// ============================================================================

// One sparse-search configuration: the batch sizes are parameters so each (metric, batch)
// combination is an independently reported case. The two sparse fixtures below share the
// n/m/dim/k dimensions and the search+check tail, but build different CSR patterns.
struct SparseParam {
  Metric metric;
  int batch_index;
  int batch_query;
};

std::ostream& operator<<(std::ostream& os, const SparseParam& p)
{
  return os << "{metric=" << metric_label(p.metric) << " batch_index=" << p.batch_index
            << " batch_query=" << p.batch_query << '}';
}

std::string sparse_name(const ::testing::TestParamInfo<SparseParam>& info)
{
  const auto& p = info.param;
  return metric_label(p.metric) + "_bi" + std::to_string(p.batch_index) + "_bq" +
         std::to_string(p.batch_query);
}

// Shared dimensions and search+check tail for the sparse fixtures. Off ::testing::Test for
// the same mixin reason as BruteForceFP64Base.
struct SparseFP64Base : BruteForceFP64Base {
  static constexpr int n = 19, m = 5, dim = kDim, k = 3;

  template <typename QueryView>
  void search_and_check(const sparse_index<double, int>& idx,
                        QueryView query,
                        const SparseParam& p,
                        const std::vector<double>& h_data,
                        const std::vector<double>& h_queries)
  {
    sparse_search_params params;
    params.batch_size_index = p.batch_index;
    params.batch_size_query = p.batch_query;
    auto ids                = raft::make_device_matrix<int, int64_t>(res, m, k);
    auto distances          = raft::make_device_matrix<double, int64_t>(res, m, k);
    search(res, params, idx, query, ids.view(), distances.view());
    check(h_data, h_queries, dim, k, p.metric, ids.data_handle(), distances.data_handle());
  }
};

// CSR pattern with an empty first row; the remaining rows hold columns {0, 1}.
class EmptyRowSparseFP64 : public SparseFP64Base, public ::testing::TestWithParam<SparseParam> {
 protected:
  SparseParam p                 = GetParam();
  std::vector<double> h_data    = zero_first_row(data(n));
  std::vector<double> h_queries = zero_first_row(queries(m));
  // Row 0 empty (ptr[0]==ptr[1]==0); rows 1.. each store 2 nonzeros.
  raft::device_vector<int, int64_t> dp = upload(empty_first_row_ptr(n));
  raft::device_vector<int, int64_t> qp = upload(empty_first_row_ptr(m));
  raft::device_vector<int, int64_t> dc = upload(two_cols(n));
  raft::device_vector<int, int64_t> qc = upload(two_cols(m));
  raft::device_vector<double, int64_t> dv =
    upload(std::vector<double>(h_data.begin() + kDim, h_data.end()));
  raft::device_vector<double, int64_t> qv =
    upload(std::vector<double>(h_queries.begin() + kDim, h_queries.end()));
  sparse_index<double, int> idx = build_index();

  // Explicit return type so this can run from the member-init list above (a deduced-return
  // helper cannot be called before it is defined).
  sparse_index<double, int> build_index()
  {
    return build(res,
                 raft::make_device_csr_matrix_view<const double>(
                   dv.data_handle(),
                   raft::make_device_compressed_structure_view<int, int, int>(
                     dp.data_handle(), dc.data_handle(), n, dim, 2 * (n - 1))),
                 p.metric,
                 3.0f);
  }

  static std::vector<double> zero_first_row(std::vector<double> v)
  {
    v[0] = v[1] = 0;
    return v;
  }
  static std::vector<int> empty_first_row_ptr(int rows)
  {
    std::vector<int> ptr(rows + 1, 0);
    for (int i = 1; i < rows; ++i) {
      ptr[i + 1] = ptr[i] + 2;
    }
    return ptr;
  }
  static std::vector<int> two_cols(int rows)
  {
    std::vector<int> cols;
    for (int i = 1; i < rows; ++i) {
      cols.insert(cols.end(), {0, 1});
    }
    return cols;
  }
  auto query_view()
  {
    return raft::make_device_csr_matrix_view<const double>(
      qv.data_handle(),
      raft::make_device_compressed_structure_view<int, int, int>(
        qp.data_handle(), qc.data_handle(), m, dim, 2 * (m - 1)));
  }
};

std::vector<SparseParam> empty_row_params()
{
  std::vector<SparseParam> v;
  for (auto metric : {Metric::L2Expanded,
                      Metric::L2Unexpanded,
                      Metric::L2SqrtExpanded,
                      Metric::L2SqrtUnexpanded,
                      Metric::L1,
                      Metric::LpUnexpanded}) {
    v.push_back({metric, 6, 2});
    v.push_back({metric, 7, 2});
    v.push_back({metric, 64, 64});
  }
  return v;
}

// L2/L1 have unique distances even for the empty query row.
TEST_P(EmptyRowSparseFP64, Search) { search_and_check(idx, query_view(), p, h_data, h_queries); }

INSTANTIATE_TEST_SUITE_P(Sparse,
                         EmptyRowSparseFP64,
                         ::testing::ValuesIn(empty_row_params()),
                         sparse_name);

// Fully dense CSR (every column present) for the similarity metrics.
class SimilaritySparseFP64 : public SparseFP64Base, public ::testing::TestWithParam<SparseParam> {
 protected:
  SparseParam p                           = GetParam();
  std::vector<double> h_data              = data(n);
  std::vector<double> h_queries           = queries(m);
  raft::device_vector<int, int64_t> dp    = upload(dense_ptr(n));
  raft::device_vector<int, int64_t> qp    = upload(dense_ptr(m));
  raft::device_vector<int, int64_t> dc    = upload(dense_cols(n));
  raft::device_vector<int, int64_t> qc    = upload(dense_cols(m));
  raft::device_vector<double, int64_t> dv = upload(h_data);
  raft::device_vector<double, int64_t> qv = upload(h_queries);
  sparse_index<double, int> idx           = build_index();

  // Explicit return type so this can run from the member-init list above (a deduced-return
  // helper cannot be called before it is defined).
  sparse_index<double, int> build_index()
  {
    return build(res,
                 raft::make_device_csr_matrix_view<const double>(
                   dv.data_handle(),
                   raft::make_device_compressed_structure_view<int, int, int>(
                     dp.data_handle(), dc.data_handle(), n, dim, n * dim)),
                 p.metric);
  }

  static std::vector<int> dense_ptr(int rows)
  {
    std::vector<int> ptr(rows + 1);
    for (int i = 0; i <= rows; ++i) {
      ptr[i] = i * dim;
    }
    return ptr;
  }
  static std::vector<int> dense_cols(int rows)
  {
    std::vector<int> cols(rows * dim);
    for (int i = 0; i < rows * dim; ++i) {
      cols[i] = i % dim;
    }
    return cols;
  }
  auto query_view()
  {
    return raft::make_device_csr_matrix_view<const double>(
      qv.data_handle(),
      raft::make_device_compressed_structure_view<int, int, int>(
        qp.data_handle(), qc.data_handle(), m, dim, m * dim));
  }
};

std::vector<SparseParam> similarity_params()
{
  std::vector<SparseParam> v;
  for (auto metric : {Metric::InnerProduct, Metric::CosineExpanded}) {
    v.push_back({metric, 6, 2});
    v.push_back({metric, 64, 2});
  }
  return v;
}

TEST_P(SimilaritySparseFP64, Search) { search_and_check(idx, query_view(), p, h_data, h_queries); }

INSTANTIATE_TEST_SUITE_P(Sparse,
                         SimilaritySparseFP64,
                         ::testing::ValuesIn(similarity_params()),
                         sparse_name);

// ============================================================================
// Serialization (public API)
// ============================================================================

// One serialize/deserialize round-trip configuration.
struct SerdeParam {
  Metric metric;
  bool include_dataset;
  bool file;
};

std::ostream& operator<<(std::ostream& os, const SerdeParam& p)
{
  return os << "{metric=" << metric_label(p.metric) << " include_dataset=" << p.include_dataset
            << " file=" << p.file << '}';
}

std::string serde_name(const ::testing::TestParamInfo<SerdeParam>& info)
{
  const auto& p = info.param;
  return metric_label(p.metric) + (p.include_dataset ? "_ds" : "_nods") +
         (p.file ? "_file" : "_stream");
}

class SerializationFP64 : public BruteForceFP64Base, public ::testing::TestWithParam<SerdeParam> {
 protected:
  static constexpr int64_t n = 17, m = 3, k = 3;
  SerdeParam p                                   = GetParam();
  std::vector<double> h_data                     = data(n);
  std::vector<double> h_queries                  = queries(m);
  raft::device_vector<double, int64_t> d_queries = upload(h_queries);
  index<double, double> idx                      = build_index();

  index<double, double> build_index()
  {
    index_params params;
    params.metric = p.metric;
    return build(
      res, params, raft::make_host_matrix_view<const double, int64_t>(h_data.data(), n, kDim));
  }
};

TEST_P(SerializationFP64, RoundTrip)
{
  index<double, double> restored(res);
  if (p.file) {
    cuvs::neighbors::tmp_index_file tmp;
    serialize(res, tmp.filename, idx, p.include_dataset);
    deserialize(res, tmp.filename, &restored);
  } else {
    std::stringstream buffer;
    serialize(res, buffer, idx, p.include_dataset);
    deserialize(res, buffer, &restored);
    buffer.seekg(0);
    index<float, float> wrong_type(res);
    EXPECT_THROW(deserialize(res, buffer, &wrong_type), raft::logic_error);
  }
  EXPECT_EQ(restored.metric(), p.metric);
  EXPECT_EQ(restored.metric_arg(), double(index_params{}.metric_arg));
  EXPECT_EQ(restored.has_norms(), idx.has_norms());
  if (!p.include_dataset) {
    restored.update_dataset(
      res, raft::make_host_matrix_view<const double, int64_t>(h_data.data(), n, kDim));
  }
  EXPECT_EQ(restored.size(), n);
  auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, m, k);
  auto distances = raft::make_device_matrix<double, int64_t>(res, m, k);
  search(res,
         search_params{},
         restored,
         raft::make_device_matrix_view<const double, int64_t>(d_queries.data_handle(), m, kDim),
         ids.view(),
         distances.view());
  check(h_data, h_queries, kDim, k, p.metric, ids.data_handle(), distances.data_handle());
}

std::vector<SerdeParam> serde_params()
{
  std::vector<SerdeParam> v;
  for (auto metric : {Metric::L2Expanded, Metric::InnerProduct}) {
    for (bool include_dataset : {false, true}) {
      for (bool file : {false, true}) {
        v.push_back({metric, include_dataset, file});
      }
    }
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(Serialization,
                         SerializationFP64,
                         ::testing::ValuesIn(serde_params()),
                         serde_name);

// ============================================================================
// Implementation-detail paths
// ============================================================================
//
// These tests reach code the public build()/search() API never routes double through, so they
// call detail:: entry points directly. They share the BruteForceFP64Base helpers but need no
// parameter grid, so they stay plain TEST_F cases on a dedicated fixture.
class BruteForceDetailFP64 : public BruteForceFP64Base, public ::testing::Test {};

TEST_F(BruteForceDetailFP64, MultipleColumnTiles)
{
  constexpr int64_t n = 129, m = 5, dim = kDim, k = 7;
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

TEST_F(BruteForceDetailFP64, ManyQueriesCrossSelectKThreshold)
{
  // rows > 198 (with k > 2 and cols <= 22061) flips raft's select_k heuristic from
  // kWarpImmediate to kWarpDistributedShm (see choose_select_k_algorithm). Pairing that
  // with a forced column tile split exercises both the select_k row threshold and the
  // column-tiling cutover for double in the same call.
  constexpr int64_t n = 129, m = 199, dim = kDim, k = 7;
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

TEST_F(BruteForceDetailFP64, MultiPartitionMerge)
{
  // The public build()/search() API always calls brute_force_knn_impl with a single dataset
  // partition, so knn_merge_parts (which combines per-partition top-k results) never runs for
  // double through that path. Call brute_force_knn_impl directly with two partitions to
  // exercise it.
  constexpr int64_t n = 129, n1 = 70, m = 5, dim = kDim, k = 7;
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

TEST_F(BruteForceDetailFP64, FusedPathNeverSelectedForDouble)
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
}  // namespace
}  // namespace cuvs::neighbors::brute_force
