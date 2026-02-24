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

#include <cuvs/cublas_v2.h>
#include <cuvs/cuda_runtime.h>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>

// Brute-force reference: dot = X * Y^T via hipBLAS SGEMM, then element-wise
// epilogue on the host.  Avoids the library's pairwise_distance API so this
// test stays independent of any CK/CUTLASS dispatch changes.

namespace {

inline auto compute_dot_matrix(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> const y)
{
  int64_t m = x.extent(0);
  int64_t n = y.extent(0);
  int64_t k = x.extent(1);

  auto dot    = raft::make_device_matrix<float, int64_t>(handle, m, n);
  auto stream = raft::resource::get_cuda_stream(handle);

  // dot (row-major m×n) = X (row-major m×k) * Y^T (row-major k×n)
  // cuBLAS is column-major, so we compute:  C_cm = Y * X^T  (n×m)
  // which, when read as row-major, gives the desired m×n dot matrix.
  float alpha = 1.0f, beta = 0.0f;
  auto cublas = raft::resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas, stream));
  RAFT_CUBLAS_TRY(cublasSgemm(cublas,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              static_cast<int>(n),
                              static_cast<int>(m),
                              static_cast<int>(k),
                              &alpha,
                              y.data_handle(),
                              static_cast<int>(k),
                              x.data_handle(),
                              static_cast<int>(k),
                              &beta,
                              dot.data_handle(),
                              static_cast<int>(n)));
  return dot;
}

}  // namespace

auto compute_reference_cosine_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> const y)
{
  int64_t m = x.extent(0);
  int64_t n = y.extent(0);
  int64_t k = x.extent(1);

  auto stream = raft::resource::get_cuda_stream(handle);

  auto norm_x = raft::make_device_vector<float, int64_t>(handle, m);
  auto norm_y = raft::make_device_vector<float, int64_t>(handle, n);
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    norm_x.data_handle(), x.data_handle(), k, m, stream, raft::sqrt_op{});
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    norm_y.data_handle(), y.data_handle(), k, n, stream, raft::sqrt_op{});

  auto dot  = compute_dot_matrix(handle, x, y);
  auto dist = raft::make_device_matrix<float, int64_t>(handle, m, n);

  // cosine_distance(i,j) = 1 - dot(i,j) / (||x_i|| * ||y_j||)
  auto dist_vec = raft::make_device_vector_view<float, int64_t>(dist.data_handle(), m * n);
  raft::linalg::map_offset(handle,
                           dist_vec,
                           [dot_ptr = dot.data_handle(),
                            nx_ptr  = norm_x.data_handle(),
                            ny_ptr  = norm_y.data_handle(),
                            n] __device__(size_t idx) {
                             int64_t i   = static_cast<int64_t>(idx) / n;
                             int64_t j   = static_cast<int64_t>(idx) % n;
                             float norms = nx_ptr[i] * ny_ptr[j];
                             float sim   = (norms > 1e-8f) ? (dot_ptr[idx] / norms) : 0.0f;
                             return 1.0f - sim;
                           });

  return dist;
}

auto compute_reference_l2_expanded_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> const y)
{
  int64_t m = x.extent(0);
  int64_t n = y.extent(0);
  int64_t k = x.extent(1);

  auto stream = raft::resource::get_cuda_stream(handle);

  auto sqnorm_x = raft::make_device_vector<float, int64_t>(handle, m);
  auto sqnorm_y = raft::make_device_vector<float, int64_t>(handle, n);
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    sqnorm_x.data_handle(), x.data_handle(), k, m, stream, raft::identity_op{});
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    sqnorm_y.data_handle(), y.data_handle(), k, n, stream, raft::identity_op{});

  auto dot  = compute_dot_matrix(handle, x, y);
  auto dist = raft::make_device_matrix<float, int64_t>(handle, m, n);

  // l2_expanded(i,j) = ||x_i||^2 + ||y_j||^2 - 2 * dot(i,j)
  auto dist_vec = raft::make_device_vector_view<float, int64_t>(dist.data_handle(), m * n);
  raft::linalg::map_offset(handle,
                           dist_vec,
                           [dot_ptr = dot.data_handle(),
                            nx_ptr  = sqnorm_x.data_handle(),
                            ny_ptr  = sqnorm_y.data_handle(),
                            n] __device__(size_t idx) {
                             int64_t i = static_cast<int64_t>(idx) / n;
                             int64_t j = static_cast<int64_t>(idx) % n;
                             return nx_ptr[i] + ny_ptr[j] - 2.0f * dot_ptr[idx];
                           });

  return dist;
}
