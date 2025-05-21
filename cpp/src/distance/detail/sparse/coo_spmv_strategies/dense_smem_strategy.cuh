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
 */
/*
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

#include "base_strategy.cuh"

#include <raft/util/cuda_dev_essentials.cuh>  // raft::ceildiv

namespace cuvs {
namespace distance {
namespace detail {
namespace sparse {

template <typename value_idx, typename value_t, int tpb>
class dense_smem_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  using smem_type   = value_t*;
  using insert_type = smem_type;
  using find_type   = smem_type;

  dense_smem_strategy(const distances_config_t<value_idx, value_t>& config_)
    : coo_spmv_strategy<value_idx, value_t, tpb>(config_)
  {
  }

  inline static int smem_per_block(int n_cols)
  {
    // (HIP/AMD) This function was using the constexpr __device__ function raft::warp_size() but
    // this in host code. raft::warp_size() will always return 64 irrespective of the target
    // architecture. Instead we use raft::host_warp_size for the current device.
    int device_id{-1};
    RAFT_CUDA_TRY(cudaGetDevice(&device_id));
    return (n_cols * sizeof(value_t)) +
           ((1024 / raft::host_warp_size(device_id)) * sizeof(value_t));
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch(value_t* out_dists,
                value_idx* coo_rows_b,
                product_f product_func,
                accum_f accum_func,
                write_f write_func,
                int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * 1024);
    auto n_blocks         = this->config.a_nrows * n_blocks_per_row;

    mask_row_it<value_idx> a_indptr(this->config.a_indptr, this->config.a_nrows);

    this->_dispatch_base(*this,
                         this->config.b_ncols,
                         a_indptr,
                         out_dists,
                         coo_rows_b,
                         product_func,
                         accum_func,
                         write_func,
                         chunk_size,
                         n_blocks,
                         n_blocks_per_row);
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t* out_dists,
                    value_idx* coo_rows_a,
                    product_f product_func,
                    accum_f accum_func,
                    write_f write_func,
                    int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * 1024);
    auto n_blocks         = this->config.b_nrows * n_blocks_per_row;

    mask_row_it<value_idx> b_indptr(this->config.b_indptr, this->config.b_nrows);

    this->_dispatch_base_rev(*this,
                             this->config.a_ncols,
                             b_indptr,
                             out_dists,
                             coo_rows_a,
                             product_func,
                             accum_func,
                             write_func,
                             chunk_size,
                             n_blocks,
                             n_blocks_per_row);
  }

  __device__ inline insert_type init_insert(smem_type cache, const value_idx& cache_size)
  {
    for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
      cache[k] = 0.0;
    }
    return cache;
  }

  __device__ inline void insert(insert_type cache, const value_idx& key, const value_t& value)
  {
    cache[key] = value;
  }

  __device__ inline find_type init_find(smem_type cache, const value_idx& cache_size)
  {
    return cache;
  }

  __device__ inline value_t find(find_type cache, const value_idx& key) { return cache[key]; }
};

}  // namespace sparse
}  // namespace detail
}  // namespace distance
}  // namespace cuvs
