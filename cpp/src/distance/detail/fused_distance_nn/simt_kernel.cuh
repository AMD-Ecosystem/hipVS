/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

// clang-format off
// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// clang-format on

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include <raft/core/kvp.hpp>              // raft::KeyValuePair
#include <raft/linalg/contractions.cuh>   // Policy

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs {
namespace distance {
namespace detail {

// TODO: specialize this function for MinAndDistanceReduceOp<int, float>
// with atomicCAS of 64 bit which will eliminate mutex and shfls
template <typename P, typename OutT, typename IdxT, typename KVPair, typename ReduceOpT>
DI void updateReducedVal(
  int* mutex, OutT* min, KVPair* val, ReduceOpT red_op, IdxT m, IdxT gridStrideY)
{
  const auto lid      = threadIdx.x % raft::WarpSize;
  const auto accrowid = threadIdx.x / P::AccThCols;

  // Update each output row in order within a warp. This will resolve hang
  // issues with pre-Volta architectures
#pragma unroll
  for (int j = 0; j < (raft::WarpSize / P::AccThCols); j++) {
    if (lid == j * P::AccThCols) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto rid = gridStrideY + accrowid + i * P::AccThRows;
        if (rid < m) {
          auto value = val[i];
          while (atomicCAS(mutex + rid, 0, 1) == 1)
            ;
          __threadfence();
          red_op(rid, min + rid, value);
          __threadfence();
          atomicCAS(mutex + rid, 1, 0);
        }
      }
    }
  }
}

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename P,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename OpT,
          typename FinalLambda>
__launch_bounds__(P::Nthreads, 2) RAFT_KERNEL fusedDistanceNNkernel(OutT* min,
                                                                    const DataT* x,
                                                                    const DataT* y,
                                                                    const DataT* xn,
                                                                    const DataT* yn,
                                                                    IdxT m,
                                                                    IdxT n,
                                                                    IdxT k,
                                                                    DataT maxVal,
                                                                    int* mutex,
                                                                    ReduceOpT redOp,
                                                                    KVPReduceOpT pairRedOp,
                                                                    OpT distance_op,
                                                                    FinalLambda fin_op)
{
// compile only if below non-ampere arch.
#if __CUDA_ARCH__ < 800
  extern __shared__ char smem[];

  typedef raft::KeyValuePair<IdxT, DataT> KVPair;
  KVPair val[P::AccRowsPerTh];
#pragma unroll
  for (int i = 0; i < P::AccRowsPerTh; ++i) {
    val[i] = {0, maxVal};
  }

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [n, pairRedOp, &val, maxVal] __device__(
                         DataT acc[P::AccRowsPerTh][P::AccColsPerTh],
                         DataT * regxn,
                         DataT * regyn,
                         IdxT gridStrideX,
                         IdxT gridStrideY) {
    KVPReduceOpT pairRed_op(pairRedOp);

    // intra thread reduce
    const auto acccolid = threadIdx.x % P::AccThCols;
    const auto accrowid = threadIdx.x / P::AccThCols;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto tmpkey = acccolid + j * P::AccThCols + gridStrideX;
        KVPair tmp  = {static_cast<IdxT>(tmpkey), acc[i][j]};
        if (tmpkey < n) {
          val[i] = pairRed_op(accrowid + i * P::AccThRows + gridStrideY, tmp, val[i]);
        }
      }
    }
  };

  auto rowEpilog_lambda =
    [m, mutex, min, pairRedOp, redOp, &val, maxVal] __device__(IdxT gridStrideY) {
      KVPReduceOpT pairRed_op(pairRedOp);
      ReduceOpT red_op(redOp);

      const auto accrowid = threadIdx.x / P::AccThCols;
      const auto lid      = raft::laneId();

    // reduce
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = P::AccThCols / 2; j > 0; j >>= 1) {
          // Actually, the srcLane (lid +j) should be (lid +j) % P:AccThCols,
          // but the shfl op applies the modulo internally.
          auto tmpkey   = raft::shfl(val[i].key, lid + j, P::AccThCols);
          auto tmpvalue = raft::shfl(val[i].value, lid + j, P::AccThCols);
          KVPair tmp    = {tmpkey, tmpvalue};
          val[i]        = pairRed_op(accrowid + i * P::AccThRows + gridStrideY, tmp, val[i]);
        }
      }

      updateReducedVal<P, OutT, IdxT, KVPair, ReduceOpT>(mutex, min, val, red_op, m, gridStrideY);

    // reset the val array.
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        val[i] = {0, maxVal};
      }
    };

  IdxT lda = k, ldb = k, ldd = n;
  constexpr bool row_major = true;
  constexpr bool write_out = false;
  PairwiseDistances<DataT,
                    DataT,  // OutT (unused in PairwiseDistances)
                    IdxT,
                    P,
                    decltype(distance_op),
                    decltype(epilog_lambda),
                    FinalLambda,
                    decltype(rowEpilog_lambda),
                    row_major,
                    write_out>
    obj(x,
        y,
        m,
        n,
        k,
        lda,
        ldb,
        ldd,
        xn,
        yn,
        nullptr,  // Output pointer
        smem,
        distance_op,
        epilog_lambda,
        fin_op,
        rowEpilog_lambda);
  obj.run();
#endif
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
