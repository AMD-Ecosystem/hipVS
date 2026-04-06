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

#include "../../distance_ops/cosine.cuh"
#include "../../distance_ops/l2_exp.cuh"
#include "../helper_structs.cuh"
#include "../simt_kernel.cuh"

#include <raft/core/kvp.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>

#include <cstddef>
#include <limits>

namespace cuvs::distance::detail {

/**
 * @brief Generic CK backend implementation for fused distance NN.
 *
 * TODO: This is a placeholder implementation that currently falls back to the SIMT kernel.
 * The actual CK implementation will use CK Tile's GEMM + custom epilogue for
 * fused distance nearest neighbor computation.
 *
 * The implementation should:
 * 1. Use CK Tile's GEMM pipeline for the matrix multiplication portion
 * 2. Implement a custom epilogue that:
 *    - Computes the distance using the provided distance_op
 *    - Performs row-wise reduction to find the minimum distance and its index
 *    - Updates the output buffer with atomic operations
 *
 * @tparam DataT Data type (float, half, etc.)
 * @tparam OutT Output type (typically KeyValuePair or distance type)
 * @tparam IdxT Index type
 * @tparam Policy Kernel policy
 * @tparam ReduceOpT Reduction operation type
 * @tparam KVPReduceOpT Key-value pair reduction operation type
 * @tparam DistanceOpT Distance operation type (e.g., l2_exp_distance_op, cosine_distance_op)
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename DistanceOpT>
void fusedDistanceNNCkImpl(OutT* min,
                           const DataT* x,
                           const DataT* y,
                           const DataT* xn,
                           const DataT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           int* workspace,
                           ReduceOpT redOp,
                           KVPReduceOpT pairRedOp,
                           DistanceOpT distance_op,
                           bool initOutBuffer,
                           cudaStream_t stream)
{
  typedef Policy P;
  dim3 blk(P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();

  if (initOutBuffer) {
    auto nblks = raft::ceildiv<int>(m, P::Nthreads);
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  raft::identity_op fin_op{};

  auto kernel = fusedDistanceNNkernel<DataT,
                                      OutT,
                                      IdxT,
                                      P,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      DistanceOpT,
                                      decltype(fin_op)>;

  constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
  dim3 grid                  = launchConfigGenerator<P>(m, n, shmemSize, kernel);

  kernel<<<grid, blk, shmemSize, stream>>>(
    min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, distance_op, fin_op);
  RAFT_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief CK backend dispatch for fused L2 NN.
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedL2NNCkDispatch(OutT* min,
                         const DataT* x,
                         const DataT* y,
                         const DataT* xn,
                         const DataT* yn,
                         IdxT m,
                         IdxT n,
                         IdxT k,
                         int* workspace,
                         ReduceOpT redOp,
                         KVPReduceOpT pairRedOp,
                         bool sqrt,
                         bool initOutBuffer,
                         cudaStream_t stream)
{
  using AccT = DataT;
  ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};
  fusedDistanceNNCkImpl<DataT, OutT, IdxT, Policy, ReduceOpT, KVPReduceOpT>(
    min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, distance_op, initOutBuffer, stream);
}

/**
 * @brief CK backend dispatch for fused Cosine NN.
 */
template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedCosineNNCkDispatch(OutT* min,
                             const DataT* x,
                             const DataT* y,
                             const DataT* xn,
                             const DataT* yn,
                             IdxT m,
                             IdxT n,
                             IdxT k,
                             int* workspace,
                             ReduceOpT redOp,
                             KVPReduceOpT pairRedOp,
                             cudaStream_t stream)
{
  using AccT = DataT;
  ops::cosine_distance_op<DataT, AccT, IdxT> distance_op{};
  fusedDistanceNNCkImpl<DataT, OutT, IdxT, Policy, ReduceOpT, KVPReduceOpT>(
    min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, distance_op, false, stream);
}

}  // namespace cuvs::distance::detail
