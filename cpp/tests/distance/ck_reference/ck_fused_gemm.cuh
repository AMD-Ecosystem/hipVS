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

// Pairwise distance kernels via CK Tile multi-D GEMM + fused epilogue.
// Supports CosineDistance and L2Expanded metrics (19_gemm_multi_d style).

#pragma once

#include "cosine_distance_op.hpp"
#include "l2_expanded_op.hpp"

#include <ck_tile/core.hpp>
#include <ck_tile/core/arch/arch.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/common.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <sstream>
#include <stdexcept>
#include <thrust/transform.h>
#include <type_traits>

namespace ck_tile {

// Tile/warp/scheduler parameters for distance GEMM.
// Warp tile M/N (16x16) follows MFMA instruction shapes for 4-byte types.
// See WarpGemmDispatcher in ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp
// and the pattern used in CK's 03_gemm example (gemm_utils.hpp).
struct GemmDistanceConfig {
  static constexpr index_t M_Tile        = 64;
  static constexpr index_t N_Tile        = 64;
  static constexpr index_t K_Tile        = 32;
  static constexpr index_t M_Warp        = 2;
  static constexpr index_t N_Warp        = 2;
  static constexpr index_t K_Warp        = 1;
  static constexpr index_t M_Warp_Tile   = 16;
  static constexpr index_t N_Warp_Tile   = 16;
  static constexpr index_t K_Warp_Tile   = 16;
  static constexpr bool DoubleSmemBuffer = false;
  static constexpr auto Scheduler        = GemmPipelineScheduler::Intrawave;
};

using GemmShape = TileGemmShape<
  sequence<GemmDistanceConfig::M_Tile, GemmDistanceConfig::N_Tile, GemmDistanceConfig::K_Tile>,
  sequence<GemmDistanceConfig::M_Warp, GemmDistanceConfig::N_Warp, GemmDistanceConfig::K_Warp>,
  sequence<GemmDistanceConfig::M_Warp_Tile,
           GemmDistanceConfig::N_Warp_Tile,
           GemmDistanceConfig::K_Warp_Tile>>;

static constexpr index_t TilePartitionerGroupNum = 8;
static constexpr index_t TilePartitionerM01      = 4;

using TilePartitioner =
  GemmSpatiallyLocalTilePartitioner<GemmShape, TilePartitionerGroupNum, TilePartitionerM01>;

// Map raft layout tags to CK Tile GEMM A/B layouts.
// Row-major inputs:  A=RowMajor (stride K), B=ColumnMajor (stride K)  -> C = X * Y^T
// Col-major inputs:  A=ColumnMajor (stride M), B=RowMajor (stride N) -> same logical op
template <typename LayoutT>
struct InputLayoutTraits;

template <>
struct InputLayoutTraits<raft::layout_c_contiguous> {
  using A_Layout = tensor_layout::gemm::RowMajor;
  using B_Layout = tensor_layout::gemm::ColumnMajor;
};

template <>
struct InputLayoutTraits<raft::layout_f_contiguous> {
  using A_Layout = tensor_layout::gemm::ColumnMajor;
  using B_Layout = tensor_layout::gemm::RowMajor;
};

template <typename DataT, typename InputLayout = raft::layout_c_contiguous>
using GemmUniversalTraits =
  TileGemmUniversalTraits<true,
                          true,
                          true,
                          GemmDistanceConfig::DoubleSmemBuffer,
                          typename InputLayoutTraits<InputLayout>::A_Layout,
                          typename InputLayoutTraits<InputLayout>::B_Layout,
                          tensor_layout::gemm::RowMajor>;

// ScalarVec=false (default): policy-computed vector sizes for DRAM loads/stores.
//   Requires K aligned to vectorSizeA/B (4 for f32, 8 for f16/bf16) and
//   N aligned to epilogue vectorSizeC (4 for f32 output).
// ScalarVec=true: forces scalar (size-1) vector loads/stores via FixedVectorSize.
//   Handles fully arbitrary M, N, K at the cost of lower memory throughput.
template <typename DataT,
          typename DistT,
          typename InputLayout = raft::layout_c_contiguous,
          bool ScalarVec       = false>
using UniversalGemmProblem = UniversalGemmPipelineProblem<DataT,
                                                          DataT,
                                                          DistT,
                                                          GemmShape,
                                                          GemmUniversalTraits<DataT, InputLayout>,
                                                          GemmDistanceConfig::Scheduler,
                                                          true,
                                                          TailNumber::Full,
                                                          element_wise::PassThrough,
                                                          element_wise::PassThrough,
                                                          DataT,
                                                          ScalarVec,
                                                          1,
                                                          1>;

template <typename DataT,
          typename DistT,
          typename InputLayout = raft::layout_c_contiguous,
          bool ScalarVec       = false>
using DistanceGemmPipeline =
  GemmPipelineAgBgCrCompV3<UniversalGemmProblem<DataT, DistT, InputLayout, ScalarVec>>;

// AMD buffer instructions (buffer_load/buffer_store) use a V# resource descriptor
// whose size field is uint32, limiting each buffer to 2^32 - 1 bytes.  When any
// tensor's element-space exceeds this, the descriptor silently wraps and accesses
// fail.  We batch along M (for D/E) and N (for B) so every buffer stays under
// the limit.
constexpr uint64_t kBufferDescriptorMaxBytes = (uint64_t{1} << 32) - 1;

template <typename DistT>
inline int64_t max_m_per_batch(int64_t n)
{
  constexpr int64_t M_Tile = GemmDistanceConfig::M_Tile;
  int64_t limit = static_cast<int64_t>(kBufferDescriptorMaxBytes / (uint64_t(n) * sizeof(DistT)));
  return std::max<int64_t>((limit / M_Tile) * M_Tile, M_Tile);
}

template <typename DataT, typename InputLayout>
inline int64_t max_n_per_batch(int64_t k, int64_t n_full)
{
  constexpr int64_t N_Tile = GemmDistanceConfig::N_Tile;
  if constexpr (std::is_same_v<InputLayout, raft::layout_c_contiguous>) {
    // Row-major B: element_space = nb * k, contiguous per batch.
    int64_t limit = static_cast<int64_t>(kBufferDescriptorMaxBytes / (uint64_t(k) * sizeof(DataT)));
    return std::max<int64_t>((limit / N_Tile) * N_Tile, N_Tile);
  } else {
    // Column-major B has stride n_full; N-batching can't reduce its element_space.
    return n_full;
  }
}

}  // namespace ck_tile

#include "ck_norm_based_standalone.cuh"

// -----------------------------------------------------------------------------
// Internal helpers and kernel launch logic.
// -----------------------------------------------------------------------------
namespace detail {

// Row norms for row-major, column norms for col-major.
template <typename InputLayout, typename DistT, typename DataT, typename NormOp, typename StreamT>
inline void compute_norms(
  DistT* out, const DataT* data, int64_t num_vecs, int64_t k, StreamT stream, NormOp op)
{
  if constexpr (std::is_same_v<InputLayout, raft::layout_c_contiguous>) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(out, data, k, num_vecs, stream, op);
  } else {
    raft::linalg::colNorm<raft::linalg::L2Norm, true>(out, data, num_vecs, k, stream, op);
  }
}

// Offset a pointer into a batch: row-major advances by offset * inner_dim, col-major by offset.
template <typename T, typename InputLayout>
inline const T* batch_ptr(const T* base, int64_t offset, int64_t inner_dim)
{
  if constexpr (std::is_same_v<InputLayout, raft::layout_c_contiguous>) {
    return base + offset * inner_dim;
  } else {
    return base + offset;
  }
}

// Launch norm-based standalone kernel for one (M, N) batch.
// Uses norm_x and norm_y directly; no D0 materialization.
template <typename FastKernel,
          typename SafeKernel,
          typename DataT,
          typename DistT,
          typename InputLayout>
inline void launch_standalone_norm_based_kernel(ck_tile::stream_config const& s,
                                                const DataT* a_ptr,
                                                const DataT* b_ptr,
                                                DistT* e_ptr,
                                                const DistT* norm_x_ptr,
                                                const DistT* norm_y_ptr,
                                                ck_tile::index_t mb,
                                                ck_tile::index_t nb,
                                                ck_tile::index_t k,
                                                ck_tile::index_t stride_A,
                                                ck_tile::index_t stride_B,
                                                ck_tile::index_t stride_E)
{
  typename FastKernel::KernelArgs kargs{a_ptr,
                                        b_ptr,
                                        e_ptr,
                                        norm_x_ptr,
                                        norm_y_ptr,
                                        mb,
                                        nb,
                                        static_cast<ck_tile::index_t>(k),
                                        stride_A,
                                        stride_B,
                                        stride_E,
                                        1 /* k_batch */};

  if (FastKernel::IsSupportedArgument(kargs)) {
    ck_tile::launch_kernel(
      s,
      ck_tile::make_kernel(
        FastKernel{}, FastKernel::GridSize(mb, nb, 1), FastKernel::BlockSize(), 0, kargs));
  } else {
    if constexpr (std::is_same_v<InputLayout, raft::layout_c_contiguous>) {
      ck_tile::launch_kernel(
        s,
        ck_tile::make_kernel(
          SafeKernel{}, SafeKernel::GridSize(mb, nb, 1), SafeKernel::BlockSize(), 0, kargs));
    } else {
      std::ostringstream oss;
      oss << "CK norm-based standalone kernel (col-major): IsSupportedArgument returned false.";
      throw std::runtime_error(oss.str());
    }
  }
}

// Batched norm-based standalone launch over M and N tiles.
// Splits work when tensors exceed buffer descriptor limits (2^32 bytes).
template <typename FastKernel,
          typename SafeKernel,
          typename DataT,
          typename DistT,
          typename InputLayout>
inline void batched_standalone_norm_based_launch(ck_tile::stream_config const& s,
                                                 const DataT* x_ptr,
                                                 const DataT* y_ptr,
                                                 DistT* dist_ptr,
                                                 const DistT* norm_x_ptr,
                                                 const DistT* norm_y_ptr,
                                                 int64_t m,
                                                 int64_t n,
                                                 int64_t k,
                                                 ck_tile::index_t stride_A,
                                                 ck_tile::index_t stride_B)
{
  const int64_t mb_max = ck_tile::max_m_per_batch<DistT>(n);
  const int64_t nb_max = ck_tile::max_n_per_batch<DataT, InputLayout>(k, n);

  for (int64_t m_off = 0; m_off < m; m_off += mb_max) {
    auto mb               = static_cast<ck_tile::index_t>(std::min(mb_max, m - m_off));
    const DataT* a_batch  = batch_ptr<DataT, InputLayout>(x_ptr, m_off, k);
    const DistT* nx_batch = norm_x_ptr + m_off;

    for (int64_t n_off = 0; n_off < n; n_off += nb_max) {
      auto nb               = static_cast<ck_tile::index_t>(std::min(nb_max, n - n_off));
      const DataT* b_batch  = batch_ptr<DataT, InputLayout>(y_ptr, n_off, k);
      const DistT* ny_batch = norm_y_ptr + n_off;
      DistT* de_batch       = dist_ptr + m_off * n + n_off;

      launch_standalone_norm_based_kernel<FastKernel, SafeKernel, DataT, DistT, InputLayout>(
        s,
        a_batch,
        b_batch,
        de_batch,
        nx_batch,
        ny_batch,
        mb,
        nb,
        static_cast<ck_tile::index_t>(k),
        stride_A,
        stride_B,
        static_cast<ck_tile::index_t>(n) /* stride_E: row stride of full output */);
    }
  }
}

// Norm-based distance: norms -> standalone kernel (no D0, no map_offset).
// Parameterized by NormOp (for compute_norms) and ElementWiseOp.
template <typename DataT,
          typename DistT,
          typename InputLayout,
          typename NormOp,
          typename ElementWiseOp>
inline auto compute_ck_norm_based_distance_impl(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, int64_t, InputLayout> const x,
  raft::device_matrix_view<const DataT, int64_t, InputLayout> const y)
{
  using FastKernel =
    ck_tile::PairwiseDistanceCkStandaloneKernel<DataT, DistT, InputLayout, false, ElementWiseOp>;
  using SafeKernel =
    ck_tile::PairwiseDistanceCkStandaloneKernel<DataT, DistT, InputLayout, true, ElementWiseOp>;

  int64_t m = x.extent(0);
  int64_t n = y.extent(0);
  int64_t k = x.extent(1);

  auto dist   = raft::make_device_matrix<DistT, int64_t>(handle, m, n);
  auto norm_x = raft::make_device_vector<DistT, int64_t>(handle, m);
  auto norm_y = raft::make_device_vector<DistT, int64_t>(handle, n);

  auto stream = raft::resource::get_cuda_stream(handle);

  compute_norms<InputLayout>(norm_x.data_handle(), x.data_handle(), m, k, stream, NormOp{});
  compute_norms<InputLayout>(norm_y.data_handle(), y.data_handle(), n, k, stream, NormOp{});

  ck_tile::index_t stride_A, stride_B;
  if constexpr (std::is_same_v<InputLayout, raft::layout_c_contiguous>) {
    stride_A = static_cast<ck_tile::index_t>(k);
    stride_B = static_cast<ck_tile::index_t>(k);
  } else {
    stride_A = static_cast<ck_tile::index_t>(m);
    stride_B = static_cast<ck_tile::index_t>(n);
  }

  ck_tile::stream_config s{stream};
  batched_standalone_norm_based_launch<FastKernel, SafeKernel, DataT, DistT, InputLayout>(
    s,
    x.data_handle(),
    y.data_handle(),
    dist.data_handle(),
    norm_x.data_handle(),
    norm_y.data_handle(),
    m,
    n,
    k,
    stride_A,
    stride_B);

  return dist;
}

}  // namespace detail

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
// Cosine: E = 1 - (x·y) / (||x|| ||y||). Norms are ||·|| (L2).
template <typename DataT, typename DistT, typename InputLayout = raft::layout_c_contiguous>
inline auto compute_ck_cosine_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, int64_t, InputLayout> const x,
  raft::device_matrix_view<const DataT, int64_t, InputLayout> const y)
{
  return detail::compute_ck_norm_based_distance_impl<DataT,
                                                     DistT,
                                                     InputLayout,
                                                     raft::sqrt_op,
                                                     ck_tile::element_wise::CosineDistance>(
    handle, x, y);
}

// L2Expanded: E = ||x||^2 + ||y||^2 - 2(x·y). Norms are ||·||^2 (squared L2).
template <typename DataT, typename DistT, typename InputLayout = raft::layout_c_contiguous>
inline auto compute_ck_l2_expanded_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, int64_t, InputLayout> const x,
  raft::device_matrix_view<const DataT, int64_t, InputLayout> const y)
{
  return detail::compute_ck_norm_based_distance_impl<DataT,
                                                     DistT,
                                                     InputLayout,
                                                     raft::identity_op,
                                                     ck_tile::element_wise::L2Expanded>(
    handle, x, y);
}
