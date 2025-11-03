/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <raft/core/operators.hpp>            // raft::abs
#include <raft/util/cuda_dev_essentials.cuh>  // DI

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
namespace cuvs::distance::detail::ops {

/**
 * @brief The canberra distance matrix calculation
 *
 * It computes the following equation:
 *
 *  c_ij = sum_k |x_ik - y_kj| / ( |x_ik| + |y_kj| )
 */
template <typename DataType, typename AccType, typename IdxType>
struct canberra_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = true;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    if constexpr ((std::is_same_v<AccT, float> && std::is_same_v<DataT, half>)) {
      AccT _x         = __half2float(x);
      AccT _y         = __half2float(y);
      const auto diff = raft::abs(_x - _y);
      const auto add  = raft::abs(_x) + raft::abs(_y);
      // deal with potential for 0 in denominator by
      // forcing 0/1 instead
      acc += ((add != 0) * diff / (add + (add == 0)));
    } else {
      const auto diff = raft::abs(x - y);
      const auto add  = raft::abs(x) + raft::abs(y);
      // deal with potential for 0 in denominator by
      // forcing 0/1 instead
      acc += ((add != 0) * diff / (add + (add == 0)));
    }
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 AccT* regxn,
                 AccT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    return;
  }
};

}  // namespace cuvs::distance::detail::ops
