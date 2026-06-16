/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cuvs/cuda_runtime.h>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, typename LutT>
struct selected {
  dim3 grid_dim;
  dim3 block_dim;
  size_t smem_size;
  size_t device_lut_size;

  selected() = default;

  selected(const void* kernel_ptr, dim3 grid, dim3 block, size_t smem, size_t device_lut)
    : grid_dim(grid),
      block_dim(block),
      smem_size(smem),
      device_lut_size(device_lut),
      kernel(kernel_ptr)
  {
  }

  // This function is ONLY defined in `ivf_pq_compute_similarity_impl.cuh` and compiled in related
  // TUs. Do not re-define it in any other TU as accessing the kernel pointer elsewhere is
  // prohibited per CUDA whole compilation rules. See
  // https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/
  template <typename O, typename L>
  friend auto get_kernel(selected<O, L> s) -> const void*;

 private:
  const void* kernel;  // Type-erased kernel pointer (compatible with any __global__ function)
};

}  // namespace cuvs::neighbors::ivf_pq::detail
