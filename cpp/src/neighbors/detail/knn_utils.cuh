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

#include <raft/core/math.hpp>

#include <cuvs/distance/distance.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#include <hip/hip_fp16.h>
#else
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#endif

namespace cuvs::neighbors::detail {

template <typename value_idx, typename value_t, typename output_t, typename expansion_f>
RAFT_KERNEL epilogue_on_csr_kernel(output_t* __restrict__ compressed_C,
                                   const value_idx* __restrict__ rows,
                                   const value_idx* __restrict__ cols,
                                   const output_t* __restrict__ Q_sq_norms,
                                   const value_t* __restrict__ R_sq_norms,
                                   value_idx nnz,
                                   expansion_f expansion_func)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;
  const value_idx i = rows[tid];
  const value_idx j = cols[tid];

  compressed_C[tid] = expansion_func(compressed_C[tid], Q_sq_norms[i], R_sq_norms[j]);
}

template <typename value_idx, typename value_t, typename output_t, int tpb = 256>
void epilogue_on_csr(raft::resources const& handle,
                     output_t* compressed_C,
                     const value_idx nnz,
                     const value_idx* rows,
                     const value_idx* cols,
                     const output_t* Q_sq_norms,
                     const value_t* R_sq_norms,
                     cuvs::distance::DistanceType metric)
{
  if (nnz == 0) return;
  auto stream = raft::resource::get_cuda_stream(handle);

  int blocks = raft::ceildiv<size_t>((size_t)nnz, tpb);
  if (metric == cuvs::distance::DistanceType::L2Expanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(output_t dot, output_t q_norm, value_t r_norm) -> output_t {
        if constexpr (std::is_same_v<value_t, half>) {
          return output_t(-2.0) * dot + q_norm + __half2float(r_norm);
        } else {
          return output_t(-2.0) * dot + q_norm + r_norm;
        }
      });
  } else if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(output_t dot, output_t q_norm, value_t r_norm) -> output_t {
        if constexpr (std::is_same_v<value_t, half>) {
          return raft::sqrt(output_t(-2.0) * dot + q_norm + __half2float(r_norm));
        } else {
          return raft::sqrt(output_t(-2.0) * dot + q_norm + r_norm);
        }
      });
  } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(output_t dot, output_t q_norm, value_t r_norm) -> output_t {
        if constexpr (std::is_same_v<value_t, half>) {
          return output_t(1.0) - dot / (q_norm * __half2float(r_norm));
        } else {
          return output_t(1.0) - dot / (q_norm * r_norm);
        }
      });
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cuvs::neighbors::detail
