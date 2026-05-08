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
#include <cstdint>
#include <raft/core/logger_macros.hpp>
#ifndef CUVS_CK_ENABLED
#error "This file requires the CK backend (CUVS_CK_ENABLED)."
#endif

// =============================================================================
// FUSED DISTANCE NEAREST NEIGHBOR - CK BACKEND DISPATCH
// =============================================================================
//
// ARCHITECTURE OVERVIEW:
// ----------------------
// This module provides GPU kernel dispatch for computing fused distance nearest
// neighbor (NN) operations. It selects between two backend implementations:
//
//   1. CK Backend (Composable Kernel): High-performance AMD GPU implementation
//      using CK Tile's GEMM pipeline with custom epilogue for distance + reduction.
//      Supports: float on compatible problem sizes. double uses SIMT fallback.
//
//   2. SIMT Fallback: Generic CUDA/HIP implementation using simple thread-level
//      parallelism. Used when CK doesn't support the data type or problem size.
//
// DATA FLOW:
// ----------
//   Inputs:  x[m,k], y[n,k] - point matrices
//            xn[m], yn[n]   - precomputed norms (||x||^2 or ||x|| depending on metric)
//
//   Compute: For each row i in x, find j = argmin_j distance(x[i], y[j])
//            Distance is computed as: epilogue(x[i] @ y[j]^T, xn[i], yn[j])
//
//   Output:  min[m] - KeyValuePair<IdxT, DataT> containing (best_j, min_distance)
//
// CONTROL FLOW:
// -------------
//   fusedL2NNCkDispatch / fusedCosineNNCkDispatch  (public API)
//                    |
//                    v
//   DistanceTraits<Metric>::dispatch(...)   (selects CK op type, handles sqrt)
//                    |
//                    v
//   dispatchWithFallback(...)               (CK vs SIMT selection)
//          |                    |
//          v                    v
//   launchCkKernel           fusedDistanceNNkernel (SIMT fallback)
//   (CK GEMM + epilogue)    (generic SIMT kernel)
//
// =============================================================================

#include "../../distance_ops/cosine.cuh"
#include "../../distance_ops/l2_exp.cuh"
#include "../../pairwise_matrix/ck/element_wise_ops.hpp"
#include "../helper_structs.cuh"
#include "../simt_kernel.cuh"
#include "fused_nn_kernel.hpp"

#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/pow2_utils.cuh>

#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>

#include <cstddef>
#include <limits>

// Buffer descriptor max bytes is 4GB - 1. The AMD GPU buffer resource descriptor (used by CK's
// raw buffer load/store) has a 32-bit range field.
constexpr uint64_t kBufferDescriptorMaxBytes = std::numeric_limits<uint32_t>::max();

namespace cuvs::distance::detail {

// =============================================================================
// BATCHING UTILITY FUNCTIONS
// =============================================================================

// Skinny-K thresholds for config selection:
//   K <= kVerySkinnyKThreshold  -> FusedDistanceNNGemmConfigVerySkinnyF32 (K_Tile = 4)
//   K <= kSkinnyKThreshold      -> FusedDistanceNNGemmConfigSkinnyF32    (K_Tile = 8)
//   K >  kSkinnyKThreshold      -> FusedDistanceNNGemmConfig             (K_Tile = 32)
constexpr uint32_t kVerySkinnyKThreshold = 8;
constexpr uint32_t kSkinnyKThreshold     = 32;

/**
 * @brief Calculate the maximum batch size for a tiled dimension.
 *
 * Constrains the data matrix (batch*k*sizeof(float)) to stay under the 4GB buffer descriptor
 * limit. When ClampOutput is true, also constrains the output buffer (batch*sizeof(OutT)).
 * Only float data type is supported for CK.
 */
template <int64_t Tile, bool ClampOutput, typename OutT>
inline int64_t max_per_batch_fused_nn(int64_t k, int64_t full_dim)
{
  if (k <= 0) return full_dim;

  // Limit based on data matrix: batch * k * sizeof(float) < 4GB
  int64_t limit = static_cast<int64_t>(kBufferDescriptorMaxBytes / (uint64_t(k) * sizeof(float)));

  if constexpr (ClampOutput) {
    // Optionally limit based on output buffer: batch * sizeof(OutT) < 4GB
    int64_t limit_out = static_cast<int64_t>(kBufferDescriptorMaxBytes / sizeof(OutT));
    limit             = std::min(limit, limit_out);
  }

  // Round down to the nearest multiple of Tile (at least one tile), cap at full_dim
  return std::min(std::max<int64_t>(raft::Pow2<Tile>::roundDown(limit), Tile), full_dim);
}

// =============================================================================
// CK BACKEND IMPLEMENTATION
// =============================================================================
//
// High-performance implementation using CK Tile's GEMM pipeline.
// Only float data type is supported. The kernel computes: C = X @ Y^T,
// then applies distance epilogue and row-wise reduction to find the
// minimum distance per row.

template <typename OutT,
          typename IdxT,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename CkDistOp,
          bool ApplySqrt,
          typename Config>
void launchCkKernelWithConfig(OutT* min,
                              float const* x,
                              float const* y,
                              float const* xn,
                              float const* yn,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              int* workspace,
                              ReduceOpT redOp,
                              KVPReduceOpT pairRedOp,
                              CkDistOp distance_op,
                              cudaStream_t stream)
{
  constexpr auto maxVal = std::numeric_limits<float>::max();

  using WrappedOp = OperatorWrapper<CkDistOp, raft::identity_op, ApplySqrt>;

  using KernelType =
    FusedDistanceNNComposableKernel<IdxT, OutT, WrappedOp, ReduceOpT, KVPReduceOpT, Config>;

  ck_tile::stream_config s{stream};

  // Calculate batching limits based on 4GB buffer descriptor constraint
  const int64_t mb_max =
    max_per_batch_fused_nn<KernelType::GemmPipeline::MPerBlock, true, OutT>(k, m);
  const int64_t nb_max =
    max_per_batch_fused_nn<KernelType::GemmPipeline::NPerBlock, false, OutT>(k, n);

  // Check that batched tiles stay within the 4GB buffer descriptor limit
  RAFT_EXPECTS(static_cast<uint64_t>(mb_max) * static_cast<uint64_t>(k) * sizeof(float) <
                   kBufferDescriptorMaxBytes &&
                 static_cast<uint64_t>(nb_max) * static_cast<uint64_t>(k) * sizeof(float) <
                   kBufferDescriptorMaxBytes,
               "Matrix dimensions exceed 4GB buffer descriptor limit");

  // Batch the computation over M and N dimensions
  for (int64_t m_off = 0; m_off < m; m_off += mb_max) {
    auto mb = static_cast<ck_tile::index_t>(std::min<int64_t>(mb_max, m - m_off));

    for (int64_t n_off = 0; n_off < n; n_off += nb_max) {
      auto nb = static_cast<ck_tile::index_t>(std::min<int64_t>(nb_max, n - n_off));

      typename KernelType::KernelArgs kargs{
        x + m_off * k,      // x[m_off:m_off+mb, :]
        y + n_off * k,      // y[n_off:n_off+nb, :]
        min + m_off,        // min[m_off:m_off+mb]
        xn + m_off,         // xn[m_off:m_off+mb]
        yn + n_off,         // yn[n_off:n_off+nb]
        workspace + m_off,  // workspace[m_off:m_off+mb]
        mb,
        nb,
        static_cast<ck_tile::index_t>(k),
        static_cast<ck_tile::index_t>(k),  // stride_A (row-major)
        static_cast<ck_tile::index_t>(k),  // stride_B (row-major)
        maxVal,
      };

      ck_tile::launch_kernel(
        s,
        ck_tile::make_kernel(
          KernelType{WrappedOp{raft::identity_op{}, distance_op}, redOp, pairRedOp},
          KernelType::GridSize(mb, nb),
          KernelType::BlockSize(),
          0,
          kargs));
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  }
}

/**
 * @brief Launch CK kernel with automatic config selection based on K dimension.
 *
 * Selects between skinny (K < 32) and default configs.
 */
template <typename OutT,
          typename IdxT,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename CkDistOp,
          bool ApplySqrt>
void launchCkKernel(OutT* min,
                    float const* x,
                    float const* y,
                    float const* xn,
                    float const* yn,
                    IdxT m,
                    IdxT n,
                    IdxT k,
                    int* workspace,
                    ReduceOpT redOp,
                    KVPReduceOpT pairRedOp,
                    CkDistOp distance_op,
                    cudaStream_t stream)
{
  if (k <= kVerySkinnyKThreshold) {
    launchCkKernelWithConfig<OutT,
                             IdxT,
                             ReduceOpT,
                             KVPReduceOpT,
                             CkDistOp,
                             ApplySqrt,
                             FusedDistanceNNGemmConfigVerySkinnyF32>(
      min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, distance_op, stream);
  } else if (k < kSkinnyKThreshold) {
    launchCkKernelWithConfig<OutT,
                             IdxT,
                             ReduceOpT,
                             KVPReduceOpT,
                             CkDistOp,
                             ApplySqrt,
                             FusedDistanceNNGemmConfigSkinnyF32>(
      min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, distance_op, stream);
  } else {
    launchCkKernelWithConfig<OutT,
                             IdxT,
                             ReduceOpT,
                             KVPReduceOpT,
                             CkDistOp,
                             ApplySqrt,
                             FusedDistanceNNGemmConfig>(
      min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, distance_op, stream);
  }
}

// =============================================================================
// UNIFIED DISPATCH WITH AUTOMATIC FALLBACK
// =============================================================================
//
// Attempts CK backend first; falls back to SIMT if unsupported.

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename CkDistOp,
          typename SimtDistOp,
          bool ApplySqrt>
void dispatchWithFallback(OutT* min,
                          DataT const* x,
                          DataT const* y,
                          DataT const* xn,
                          DataT const* yn,
                          IdxT m,
                          IdxT n,
                          IdxT k,
                          int* workspace,
                          ReduceOpT redOp,
                          KVPReduceOpT pairRedOp,
                          CkDistOp ck_op,
                          SimtDistOp simt_op,
                          cudaStream_t stream)
{
  if constexpr (isSupportedFusedNN<DataT>()) {
    // CK's MFMA-based intrinsics is only available on gfx9
    // (CDNA). On gfx11 (RDNA3) and other non-gfx9 architectures the CK kernel silently
    // produces wrong results, so gate on gfx9 the same way pairwise_matrix/dispatch_ck.cuh does.
    // As of the CK shipped with ROCm 7.2 in warp_gemm_dispatcher.hpp, the fp32 WarpGemmDispatcher
    // specializations for 16×16×4 and 16×16×16 are unconditionally mapped to
    // WarpGemmMfmaF32F32F32M16N16K4 / WarpGemmMfmaF32F32F32M16N16K16 — there is no #if
    // defined(__gfx11__) || defined(__gfx12__) branch for float (unlike fp16/bf16 for 16×16×16).
    int device = 0;
    RAFT_CUDA_TRY(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    RAFT_CUDA_TRY(hipGetDeviceProperties(&prop, device));
    const bool is_gfx9 = (std::strncmp(prop.gcnArchName, "gfx9", 4) == 0);

    if (is_gfx9) {
      launchCkKernel<OutT, IdxT, ReduceOpT, KVPReduceOpT, CkDistOp, ApplySqrt>(
        min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, ck_op, stream);
      return;
    }
  }

  RAFT_LOG_INFO(
    "Falling back to SIMT kernel since CK does not support the current input configuration");

  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  raft::identity_op fin_op{};

  auto kernel = fusedDistanceNNkernel<DataT,
                                      OutT,
                                      IdxT,
                                      Policy,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      SimtDistOp,
                                      decltype(fin_op)>;

  constexpr size_t shmemSize = Policy::SmemSize + ((Policy::Mblk + Policy::Nblk) * sizeof(DataT));
  dim3 grid                  = launchConfigGenerator<Policy>(m, n, shmemSize, kernel);
  dim3 blk(Policy::Nthreads);

  kernel<<<grid, blk, shmemSize, stream>>>(
    min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, simt_op, fin_op);
  RAFT_CUDA_TRY(cudaGetLastError());
}

// =============================================================================
// DISTANCE METRIC TRAITS
// =============================================================================
//
// Encapsulates metric-specific types and dispatch logic. This design allows
// adding new distance metrics without modifying the core dispatch machinery.

template <typename DataT, typename IdxT>
struct L2DistanceTraits {
  using CkOp   = ops::l2_exp_ck_op;
  using AccT   = DataT;
  using SimtOp = ops::l2_exp_distance_op<DataT, AccT, IdxT>;

  static CkOp makeCkOp(bool /* sqrt */) { return CkOp{}; }
  static SimtOp makeSimtOp(bool sqrt) { return SimtOp{sqrt}; }

  template <typename OutT,
            typename Policy,
            typename ReduceOpT,
            typename KVPReduceOpT,
            bool ApplySqrt>
  static void dispatch(OutT* min,
                       DataT const* x,
                       DataT const* y,
                       DataT const* xn,
                       DataT const* yn,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       int* workspace,
                       ReduceOpT redOp,
                       KVPReduceOpT pairRedOp,
                       cudaStream_t stream)
  {
    dispatchWithFallback<DataT,
                         OutT,
                         IdxT,
                         Policy,
                         ReduceOpT,
                         KVPReduceOpT,
                         CkOp,
                         SimtOp,
                         ApplySqrt>(min,
                                    x,
                                    y,
                                    xn,
                                    yn,
                                    m,
                                    n,
                                    k,
                                    workspace,
                                    redOp,
                                    pairRedOp,
                                    makeCkOp(ApplySqrt),
                                    makeSimtOp(ApplySqrt),
                                    stream);
  }
};

template <typename DataT, typename IdxT>
struct CosineDistanceTraits {
  using CkOp   = ops::cosine_ck_op;
  using AccT   = DataT;
  using SimtOp = ops::cosine_distance_op<DataT, AccT, IdxT>;

  static CkOp makeCkOp() { return CkOp{}; }
  static SimtOp makeSimtOp() { return SimtOp{}; }

  template <typename OutT, typename Policy, typename ReduceOpT, typename KVPReduceOpT>
  static void dispatch(OutT* min,
                       DataT const* x,
                       DataT const* y,
                       DataT const* xn,
                       DataT const* yn,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       int* workspace,
                       ReduceOpT redOp,
                       KVPReduceOpT pairRedOp,
                       cudaStream_t stream)
  {
    dispatchWithFallback<DataT, OutT, IdxT, Policy, ReduceOpT, KVPReduceOpT, CkOp, SimtOp, false>(
      min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, makeCkOp(), makeSimtOp(), stream);
  }
};

// =============================================================================
// PUBLIC API
// =============================================================================

/**
 * @brief Dispatch fused L2 nearest neighbor computation.
 *
 * For each query point x[i], finds the data point y[j] with minimum L2 distance.
 * Uses CK backend when supported, falls back to SIMT otherwise.
 *
 * @param min         Output: KeyValuePair(best_index, min_distance) for each query
 * @param x           Query points matrix [m x k], row-major
 * @param y           Data points matrix [n x k], row-major
 * @param xn          Precomputed ||x[i]||^2 norms [m]
 * @param yn          Precomputed ||y[j]||^2 norms [n]
 * @param m           Number of query points
 * @param n           Number of data points
 * @param k           Dimension of points
 * @param workspace   Mutex workspace for atomic updates [m integers]
 * @param redOp       Reduction operator for output updates
 * @param pairRedOp   Key-value pair reduction operator
 * @param sqrt        If true, return sqrt(L2) instead of L2
 * @param stream      CUDA stream
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedL2NNCkDispatch(OutT* min,
                         DataT const* x,
                         DataT const* y,
                         DataT const* xn,
                         DataT const* yn,
                         IdxT m,
                         IdxT n,
                         IdxT k,
                         int* workspace,
                         ReduceOpT redOp,
                         KVPReduceOpT pairRedOp,
                         bool sqrt,
                         cudaStream_t stream)
{
  using Traits = L2DistanceTraits<DataT, IdxT>;

  if (sqrt) {
    Traits::template dispatch<OutT, Policy, ReduceOpT, KVPReduceOpT, true>(
      min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, stream);
  } else {
    Traits::template dispatch<OutT, Policy, ReduceOpT, KVPReduceOpT, false>(
      min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, stream);
  }
}

/**
 * @brief Dispatch fused Cosine nearest neighbor computation.
 *
 * For each query point x[i], finds the data point y[j] with minimum cosine distance.
 * Cosine distance = 1 - (x·y)/(||x|| ||y||)
 *
 * @param min         Output: KeyValuePair(best_index, min_distance) for each query
 * @param x           Query points matrix [m x k], row-major
 * @param y           Data points matrix [n x k], row-major
 * @param xn          Precomputed ||x[i]|| norms [m]
 * @param yn          Precomputed ||y[j]|| norms [n]
 * @param m           Number of query points
 * @param n           Number of data points
 * @param k           Dimension of points
 * @param workspace   Mutex workspace for atomic updates [m integers]
 * @param redOp       Reduction operator for output updates
 * @param pairRedOp   Key-value pair reduction operator
 * @param stream      CUDA stream
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedCosineNNCkDispatch(OutT* min,
                             DataT const* x,
                             DataT const* y,
                             DataT const* xn,
                             DataT const* yn,
                             IdxT m,
                             IdxT n,
                             IdxT k,
                             int* workspace,
                             ReduceOpT redOp,
                             KVPReduceOpT pairRedOp,
                             cudaStream_t stream)
{
  using Traits = CosineDistanceTraits<DataT, IdxT>;
  Traits::template dispatch<OutT, Policy, ReduceOpT, KVPReduceOpT>(
    min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, stream);
}

}  // namespace cuvs::distance::detail
