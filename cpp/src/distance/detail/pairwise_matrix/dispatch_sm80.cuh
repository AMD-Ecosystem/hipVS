/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../pairwise_distance_cutlass_base.cuh"  // cutlassDistanceKernel
#include "dispatch_layout.cuh"                    // dispatch_layout

#include <algorithm>  // std::min

namespace cuvs::distance::detail {

template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(OpT distance_op,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                   SM_compat_t sm_compat_range,
                                   cudaStream_t stream)
{
  int vec_len = determine_vec_len(params);

  // f takes compile-time constants row_major and vec_len aligned and runs the
  // corresponding cutlass launch code.
  auto f = [&](auto row_major, auto vec_len_aligned) {
    // row_major and vec_len are std::integral_constants of type bool and int
    // respectively.

    // Prevent double, vec_len=4 combination (this is not supported)
    constexpr int vec_len = std::min(vec_len_aligned(), static_cast<int>(16 / sizeof(DataT)));

    using AccT = typename OpT::AccT;
    cutlassDistanceKernel<DataT, AccT, OutT, IdxT, vec_len, FinOpT, OpT, row_major()>(params.x,
                                                                                      params.y,
                                                                                      params.x_norm,
                                                                                      params.y_norm,
                                                                                      params.m,
                                                                                      params.n,
                                                                                      params.k,
                                                                                      params.ldx,
                                                                                      params.ldy,
                                                                                      params.ld_out,
                                                                                      params.out,
                                                                                      params.fin_op,
                                                                                      distance_op,
                                                                                      stream);
  };

  // Dispatch_layout calls f with appropriate compile time constants based on
  // the runtime values of params.is_row_major and vec_len.
  dispatch_layout(params.is_row_major, vec_len, f);
}

};  // namespace cuvs::distance::detail
