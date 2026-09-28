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
 *
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

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cusparse.h>
#else
#include <cusparse_v2.h>
#endif
#include <gtest/gtest.h>

namespace cuvs {
namespace neighbors {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct SparseKNNInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;
  std::vector<value_idx> out_indices_ref_h;

  int k;

  int batch_size_index = 2;
  int batch_size_query = 2;

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded;
  float metric_arg                    = 3.0f;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os, const SparseKNNInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class SparseKNNTest : public ::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>> {
 public:
  SparseKNNTest()
    : params(::testing::TestWithParam<SparseKNNInputs<value_idx, value_t>>::GetParam()),
      indptr(0, resource::get_cuda_stream(handle)),
      indices(0, resource::get_cuda_stream(handle)),
      data(0, resource::get_cuda_stream(handle)),
      out_indices(0, resource::get_cuda_stream(handle)),
      out_dists(0, resource::get_cuda_stream(handle)),
      out_indices_ref(0, resource::get_cuda_stream(handle)),
      out_dists_ref(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override
  {
    n_rows = params.indptr_h.size() - 1;
    nnz    = params.indices_h.size();
    k      = params.k;

    make_data();

    auto index_structure =
      raft::make_device_compressed_structure_view<value_idx, value_idx, value_idx>(
        indptr.data(), indices.data(), n_rows, params.n_cols, nnz);
    auto index_csr = raft::make_device_csr_matrix_view<const value_t>(data.data(), index_structure);

    auto index =
      cuvs::neighbors::brute_force::build(handle, index_csr, params.metric, params.metric_arg);

    cuvs::neighbors::brute_force::sparse_search_params search_params;
    search_params.batch_size_index = params.batch_size_index;
    search_params.batch_size_query = params.batch_size_query;

    cuvs::neighbors::brute_force::search(
      handle,
      search_params,
      index,
      index_csr,
      raft::make_device_matrix_view<value_idx, int64_t>(out_indices.data(), n_rows, k),
      raft::make_device_matrix_view<value_t, int64_t>(out_dists.data(), n_rows, k));

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));
  }

  void compare()
  {
    ASSERT_TRUE(devArrMatch(
      out_dists_ref.data(), out_dists.data(), n_rows * k, CompareApprox<value_t>(1e-4)));
    ASSERT_TRUE(
      devArrMatch(out_indices_ref.data(), out_indices.data(), n_rows * k, Compare<value_idx>()));
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    auto stream = resource::get_cuda_stream(handle);
    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);

    std::vector<value_t> out_dists_ref_h     = params.out_dists_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;

    out_indices_ref.resize(out_indices_ref_h.size(), stream);
    out_dists_ref.resize(out_dists_ref_h.size(), stream);

    update_device(
      out_indices_ref.data(), out_indices_ref_h.data(), out_indices_ref_h.size(), stream);
    update_device(out_dists_ref.data(), out_dists_ref_h.data(), out_dists_ref_h.size(), stream);

    out_dists.resize(n_rows * k, stream);
    out_indices.resize(n_rows * k, stream);
  }

  raft::resources handle;

  int n_rows, nnz, k;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  // output data
  rmm::device_uvector<value_idx> out_indices;
  rmm::device_uvector<value_t> out_dists;

  rmm::device_uvector<value_idx> out_indices_ref;
  rmm::device_uvector<value_t> out_dists_ref;

  SparseKNNInputs<value_idx, value_t> params;
};

const std::vector<SparseKNNInputs<int, float>> inputs_i32_f = {
  {9,                                                 // ncols
   {0, 2, 4, 6, 8},                                   // indptr
   {0, 4, 0, 3, 0, 2, 0, 8},                          // indices
   {0.0f, 1.0f, 5.0f, 6.0f, 5.0f, 6.0f, 0.0f, 1.0f},  // data
   {0, 1.41421, 0, 7.87401, 0, 7.87401, 0, 1.41421},  // dists
   {0, 3, 1, 0, 2, 0, 3, 0},                          // inds
   2,
   2,
   2,
   cuvs::distance::DistanceType::L2SqrtExpanded},
  // Similarity merging must keep maxima; the final batch has fewer than k rows.
  {1,
   {0, 1, 2, 3, 4, 5},
   {0, 0, 0, 0, 0},
   {5, 1, 3, 2, 4},
   {25, 20, 5, 4, 15, 12, 10, 8, 20, 16},
   {0, 4, 0, 4, 0, 4, 0, 4, 0, 4},
   2,
   2,
   2,
   cuvs::distance::DistanceType::InnerProduct},
  // Odd Minkowski exponents must apply to absolute differences. The leading element is an
  // explicitly stored zero on purpose: the unexpanded metrics split the column union across a
  // forward and a reverse spmv pass, and a stored zero must not be mistaken for a column absent
  // from the other row and counted by both.
  {1,
   {0, 1, 2, 3, 4, 5},
   {0, 0, 0, 0, 0},
   {0, 2, 5, 9, 14},
   {0, 2, 0, 2, 0, 3, 0, 4, 0, 5},
   {0, 1, 1, 0, 2, 1, 3, 2, 4, 3},
   2,
   2,
   2,
   cuvs::distance::DistanceType::LpUnexpanded},
  // As above, but the stored zero is negative. -0.0 is the marker the spmv kernel uses for an
  // absent column, so the strategies must normalize it on insert to keep the two apart. Negating
  // zero does not change any distance.
  {1,
   {0, 1, 2, 3, 4, 5},
   {0, 0, 0, 0, 0},
   {-0.0f, 2, 5, 9, 14},
   {0, 2, 0, 2, 0, 3, 0, 4, 0, 5},
   {0, 1, 1, 0, 2, 1, 3, 2, 4, 3},
   2,
   2,
   2,
   cuvs::distance::DistanceType::LpUnexpanded},
  // Both of the above again, but wide enough (> the dense shared-memory column budget) that the
  // spmv falls onto the hash strategy, which marks absent columns in its own way. Row 2 occupies
  // a column the other rows lack, so the reverse pass has to distinguish a genuinely absent
  // column from row 0's stored zero rather than skipping both. k covers every pair: without the
  // reverse contribution d(1,2) comes out as 64^(1/3) = 4 instead of 72^(1/3).
  {20000,
   {0, 1, 2, 3},
   {0, 0, 5000},
   {-0.0f, 2, 4},
   {0, 2, 4, 0, 2, 4.1601676, 0, 4, 4.1601676},
   {0, 1, 2, 1, 0, 2, 2, 0, 1},
   3,
   3,
   3,
   cuvs::distance::DistanceType::LpUnexpanded}};
typedef SparseKNNTest<int, float> SparseKNNTestF;
TEST_P(SparseKNNTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseKNNTest, SparseKNNTestF, ::testing::ValuesIn(inputs_i32_f));

};  // end namespace neighbors
};  // end namespace cuvs
