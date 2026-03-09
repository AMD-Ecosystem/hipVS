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

#pragma once
#ifndef CUVS_CK_ENABLED
#error "This file requires the CK backend (CUVS_CK_ENABLED)."
#endif

#include "dispatch_layout.cuh"

#include <ck_tile/core.hpp>
#include <ck_tile/core/arch/arch.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/common.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <cuvs/cuda_runtime.h>
#include <hip/hip_fp16.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <raft/core/math.hpp>

#include "../distance_ops/cosine.cuh"  // ops::cosine_ck_op
#include "../distance_ops/l2_exp.cuh"  // ops::l2_exp_ck_op

#include <cassert>

#include "ck/ck_kernel.hpp"

namespace cuvs::distance::detail {

template <typename T, typename = void>
struct has_sqrt_member : std::false_type {};

template <typename T>
struct has_sqrt_member<T, std::void_t<decltype(std::declval<T>().sqrt)>> : std::true_type {};

// Buffer descriptor max bytes is 4GB - 1. The AMD GPU buffer resource descriptor (used by CK's
// raw buffer load/store) has a 32-bit range field. See ck_tile/core/arch/amd_buffer_addressing.hpp:
// struct buffer_resource { const void* ptr; uint32_t range; uint32_t config; } and
// make_wave_buffer_resource(ptr, uint32_t size = 0xffffffff).
constexpr uint64_t kBufferDescriptorMaxBytes = std::numeric_limits<uint32_t>::max();

/**
 * @brief Calculate the maximum batch size for M.
 *
 * Constrains both output (mb*nb*sizeof(OutT)) and A matrix (mb*k*sizeof(DataT)) to stay under
 * the 4GB buffer descriptor limit.
 *
 * @tparam DataT
 * @tparam OutT
 * @param n
 * @param k
 * @return int64_t
 */
template <typename DataT, typename OutT>
inline int64_t max_m_per_batch(int64_t n, int64_t k)
{
  constexpr int64_t M_Tile = GemmDistanceConfig::M_Tile;
  if (n <= 0 || k <= 0) return M_Tile;
  int64_t limit_out =
    static_cast<int64_t>(kBufferDescriptorMaxBytes / (uint64_t(n) * sizeof(OutT)));
  int64_t limit_a = static_cast<int64_t>(kBufferDescriptorMaxBytes / (uint64_t(k) * sizeof(DataT)));
  int64_t limit   = std::min(limit_out, limit_a);
  return std::max<int64_t>((limit / M_Tile) * M_Tile, M_Tile);
}

/**
 * @brief Calculate the maximum batch size for N.
 *
 * @tparam DataT
 * @param k
 * @param n_full
 * @param is_row_major
 * @return int64_t
 */
template <typename DataT>
int64_t max_n_per_batch(int64_t k, int64_t n_full, bool is_row_major)
{
  constexpr int64_t N_Tile = GemmDistanceConfig::N_Tile;
  if (is_row_major) {
    int64_t limit = static_cast<int64_t>(kBufferDescriptorMaxBytes / (uint64_t(k) * sizeof(DataT)));
    return std::max<int64_t>((limit / N_Tile) * N_Tile, N_Tile);
  } else {
    return n_full;
  }
}

template <typename KernelT, typename KArgsT>
void launch_ck_gemm(const ck_tile::stream_config& s,
                    const KArgsT& kargs,
                    ck_tile::index_t m,
                    ck_tile::index_t n)
{
  ck_tile::launch_kernel(
    s, ck_tile::make_kernel(KernelT{}, KernelT::GridSize(m, n, 1), KernelT::BlockSize(), 0, kargs));
}

template <typename FastKernel,
          typename SafeKernel,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename APtrFn,
          typename BPtrFn,
          typename NormXFn,
          typename NormYFn>
void dispatch_batched_pairwise_distance_ck_gemm(
  const pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>& params,
  int64_t mb_max,
  int64_t nb_max,
  const ck_tile::stream_config& s,
  APtrFn&& a_ptr_fn,
  BPtrFn&& b_ptr_fn,
  NormXFn&& norm_x_fn,
  NormYFn&& norm_y_fn)
{
  const auto m = static_cast<int64_t>(params.m);
  const auto n = static_cast<int64_t>(params.n);

  for (int64_t m_off = 0; m_off < m; m_off += mb_max) {
    auto mb = static_cast<ck_tile::index_t>(std::min(mb_max, m - m_off));
    for (int64_t n_off = 0; n_off < n; n_off += nb_max) {
      auto nb = static_cast<ck_tile::index_t>(std::min(nb_max, n - n_off));
      typename FastKernel::KernelArgs kargs{a_ptr_fn(m_off),
                                            b_ptr_fn(n_off),
                                            params.out + m_off * params.ld_out + n_off,
                                            norm_x_fn(m_off),
                                            norm_y_fn(n_off),
                                            mb,
                                            nb,
                                            static_cast<ck_tile::index_t>(params.k),
                                            static_cast<ck_tile::index_t>(params.ldx),
                                            static_cast<ck_tile::index_t>(params.ldy),
                                            static_cast<ck_tile::index_t>(params.ld_out),
                                            1};

      if (FastKernel::IsSupportedArgument(kargs)) {
        launch_ck_gemm<FastKernel>(s, kargs, mb, nb);
      } else {
        launch_ck_gemm<SafeKernel>(s, kargs, mb, nb);
      }
    }
  }
}

/**
 * @brief Internal function for dispatching the pairwise matrix operation using the CK backend.
 *
 * @tparam OpT: Type of distance operation
 * @tparam IdxT: Index type
 * @tparam DataT: Data type
 * @tparam OutT: Output type
 * @tparam FinOpT: Final operation type
 * @tparam SM_compat_t: Type of the SM architecture compatibility(unused in CK)
 * @tparam ApplySqrtToDistances: Whether to apply the sqrt operation to the distances
 * @param distance_op: Distance operation
 * @param params: Parameters
 * @param compat_range: Which SM architectures to compile for.(unused in CK)
 * @param stream: Stream
 */
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t,
          bool ApplySqrtToDistances>
void pairwise_matrix_ck_dispatch_internal(OpT distance_op,
                                          pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                          SM_compat_t compat_range,
                                          cudaStream_t stream)
{
  if constexpr (not isSupported<DataT>()) {
    // If the data type is not supported, fall back to the SM60 dispatch.
    pairwise_matrix_sm60_dispatch(distance_op, params, compat_range, stream);
  } else {
    // Map the data type to the CK data type.
    using CkDataT = CkDataType<DataT>;
    pairwise_matrix_params<IdxT, CkDataT, OutT, FinOpT> ck_params{
      params.m,
      params.n,
      params.k,
      params.ldx,
      params.ldy,
      params.ld_out,
      reinterpret_cast<const CkDataT*>(
        params.x),  // CK kernel uses ck_tile::fp16_t for half data type.
      reinterpret_cast<const CkDataT*>(params.y),
      params.x_norm,
      params.y_norm,
      params.out,
      params.fin_op,
      params.is_row_major};

    auto ck_op = distance_op.get_ck_op();

    // Hardware limitation: CK float MFMA is only implemented for gfx9.
    int device = 0;
    RAFT_CUDA_TRY(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    RAFT_CUDA_TRY(hipGetDeviceProperties(&prop, device));
    if (std::strncmp(prop.gcnArchName, "gfx9", 4) != 0) {
      if constexpr (std::is_same_v<DataT, float>) {
        pairwise_matrix_sm60_dispatch(distance_op, params, compat_range, stream);
        return;
      }
      if (!params.is_row_major) {
        // CK's WMMA path is not supported for column-major.
        pairwise_matrix_sm60_dispatch(distance_op, params, compat_range, stream);
        return;
      }
    }

    // The fast kernel requires that the input matrices are aligned to the vector load size.
    // If the input matrices are not aligned, we fall back to the safe kernel.

    using CkOpT         = decltype(ck_op);
    using EpilogueOp    = OperatorWrapper<CkOpT, FinOpT, ApplySqrtToDistances>;
    using FastKernel    = PairwiseDistanceCkKernel<CkDataT,
                                                   OutT,
                                                   ck_tile::tensor_layout::gemm::RowMajor,
                                                   false,
                                                   EpilogueOp>;
    using SafeKernel    = PairwiseDistanceCkKernel<CkDataT,
                                                   OutT,
                                                   ck_tile::tensor_layout::gemm::RowMajor,
                                                   true,
                                                   EpilogueOp>;
    using FastKernelCol = PairwiseDistanceCkKernel<CkDataT,
                                                   OutT,
                                                   ck_tile::tensor_layout::gemm::ColumnMajor,
                                                   false,
                                                   EpilogueOp>;

    // Column-major: probe first and fall back to SM60 if CK can't handle the shape.
    if (!ck_params.is_row_major) {
      PairwiseDistanceCkKernelArgs<CkDataT, OutT> probe{
        ck_params.x,
        ck_params.y,
        ck_params.out,
        ck_params.x_norm,
        ck_params.y_norm,
        static_cast<ck_tile::index_t>(ck_params.m),
        static_cast<ck_tile::index_t>(ck_params.n),
        static_cast<ck_tile::index_t>(ck_params.k),
        static_cast<ck_tile::index_t>(ck_params.ldx),
        static_cast<ck_tile::index_t>(ck_params.ldy),
        static_cast<ck_tile::index_t>(ck_params.ld_out),
        1};

      if (!FastKernelCol::IsSupportedArgument(probe)) {
        pairwise_matrix_sm60_dispatch(distance_op, params, compat_range, stream);
        return;
      }
    }

    ck_tile::stream_config s{stream};
    const auto n = static_cast<int64_t>(ck_params.n);
    const auto k = static_cast<int64_t>(ck_params.k);
    // We're forced to batch since CK uses uint32_t for the buffer descriptor size.
    const int64_t mb_max = max_m_per_batch<CkDataT, OutT>(n, k);
    const int64_t nb_max = max_n_per_batch<CkDataT>(k, n, ck_params.is_row_major);

    // Output batch mb_max * nb_max must fit in 4GB. When limit_out=0 we still return M_Tile,
    // but mb_max * nb_max * sizeof(OutT) can exceed 4GB; fall back to SM60 in that case.
    if (static_cast<uint64_t>(mb_max) * static_cast<uint64_t>(nb_max) * sizeof(OutT) >=
        kBufferDescriptorMaxBytes) {
      pairwise_matrix_sm60_dispatch(distance_op, params, compat_range, stream);
      return;
    }

    if (ck_params.is_row_major) {
      dispatch_batched_pairwise_distance_ck_gemm<FastKernel, SafeKernel>(
        ck_params,
        mb_max,
        nb_max,
        s,
        [&](int64_t off) { return ck_params.x + off * ck_params.ldx; },
        [&](int64_t off) { return ck_params.y + off * ck_params.ldy; },
        [&](int64_t off) { return ck_params.x_norm + off; },
        [&](int64_t off) { return ck_params.y_norm + off; });
    } else {
      // Column-major: SafeKernelCol (ScalarVec=true) has CK tile compatibility issues.
      // Use FastKernelCol only; when it fails for a batch, fall back to SM60 for the whole op.
      PairwiseDistanceCkKernelArgs<CkDataT, OutT> first_batch{
        ck_params.x,
        ck_params.y,
        ck_params.out,
        ck_params.x_norm,
        ck_params.y_norm,
        static_cast<ck_tile::index_t>(std::min(mb_max, static_cast<int64_t>(ck_params.m))),
        static_cast<ck_tile::index_t>(std::min(nb_max, static_cast<int64_t>(ck_params.n))),
        static_cast<ck_tile::index_t>(ck_params.k),
        static_cast<ck_tile::index_t>(ck_params.ldx),
        static_cast<ck_tile::index_t>(ck_params.ldy),
        static_cast<ck_tile::index_t>(ck_params.ld_out),
        1};
      if (!FastKernelCol::IsSupportedArgument(first_batch)) {
        pairwise_matrix_sm60_dispatch(distance_op, params, compat_range, stream);
        return;
      }
      dispatch_batched_pairwise_distance_ck_gemm<FastKernelCol, FastKernelCol>(
        ck_params,
        mb_max,
        nb_max,
        s,
        [&](int64_t off) { return ck_params.x + off; },
        [&](int64_t off) { return ck_params.y + off; },
        [&](int64_t off) { return ck_params.x_norm + off; },
        [&](int64_t off) { return ck_params.y_norm + off; });
    }
  }
}

/**
 * @brief Dispatch the pairwise matrix operation using the CK backend.
 *
 * @tparam OpT: Type of distance operation
 * @tparam IdxT: Index type
 * @tparam DataT: Data type
 * @tparam OutT: Output type
 * @tparam FinOpT: Final operation type
 * @tparam SM_compat_t: Type of the SM architecture compatibility(unused in CK)
 * @param distance_op: Distance operation
 * @param params: Parameters
 * @param compat_range: Which SM architectures to compile for.(unused in CK)
 * @param stream: Stream
 */
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_ck_dispatch(OpT distance_op,
                                 pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                 SM_compat_t compat_range,
                                 cudaStream_t stream)
{
  if constexpr (has_sqrt_member<OpT>::value) {
    if (distance_op.sqrt) {
      static constexpr bool ApplySqrtToDistances = true;
      pairwise_matrix_ck_dispatch_internal<OpT,
                                           IdxT,
                                           DataT,
                                           OutT,
                                           FinOpT,
                                           SM_compat_t,
                                           ApplySqrtToDistances>(
        distance_op, params, compat_range, stream);
      return;
    }
  }
  static constexpr bool ApplySqrtToDistances = false;
  pairwise_matrix_ck_dispatch_internal<OpT,
                                       IdxT,
                                       DataT,
                                       OutT,
                                       FinOpT,
                                       SM_compat_t,
                                       ApplySqrtToDistances>(
    distance_op, params, compat_range, stream);
}

}  // namespace cuvs::distance::detail
