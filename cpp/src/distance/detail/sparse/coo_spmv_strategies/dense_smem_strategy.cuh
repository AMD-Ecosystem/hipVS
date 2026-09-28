/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#pragma once

#include "base_strategy.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

namespace cuvs {
namespace distance {
namespace detail {
namespace sparse {

template <typename value_idx, typename value_t, int tpb>
class dense_smem_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  using map_type = value_t*;

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

  __device__ inline map_type init_map(void* storage, const value_idx& cache_size)
  {
    auto cache = static_cast<value_t*>(storage);
    for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
      // -0.0 marks "column absent from this row". It is arithmetically identical to 0.0, so it
      // leaves every product/accum op below unchanged, but is_absent() can still tell it apart
      // from a stored zero. See coo_spmv_kernel.cuh.
      cache[k] = -0.0;
    }
    return cache;
  }

  __device__ inline void insert(map_type& cache, const value_idx& key, const value_t& value)
  {
    // Adding zero maps a stored -0.0 to +0.0 and is an exact no-op for every other value, so a
    // stored zero can never be mistaken for the absent marker. Do not fold this away: it is only
    // a no-op under -ffast-math/-fno-signed-zeros, which coo_spmv_kernel.cuh rejects.
    cache[key] = value + value_t{0};
  }

  __device__ inline value_t find(map_type& cache, const value_idx& key) { return cache[key]; }
};

}  // namespace sparse
}  // namespace detail
}  // namespace distance
}  // namespace cuvs
