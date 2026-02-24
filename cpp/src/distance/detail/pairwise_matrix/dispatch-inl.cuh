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

/* This file has two responsibilities:
 *
 * 1. Dispatch to the correct implementation of a kernel based on the
 *    architecture of the device on which the kernel will be launched. For
 *    instance, the cosine distance has a CUTLASS-based implementation that can
 *    be used on SM80+ and the normal implementation that is used on older
 *    architectures.
 *
 * 2. Provide concise function templates that can be instantiated in
 *    src/distance/detail/pairwise_matrix/. Previously,
 *    cuvs::distance::detail::distance was instantiated. The function
 *    necessarily required a large set of include files, which slowed down the
 *    build. The cuvs::distance::detail::pairwise_matrix_arch_dispatch functions
 *    do not require as large an include files set, which speeds up the build.
 */

#ifdef CUVS_CK_ENABLED
#include "../distance_ops/ck.cuh"  // ops::has_ck_op
#endif
#include "../distance_ops/cutlass.cuh"           // ops::has_cutlass_op
#include "../pairwise_matrix/dispatch_sm60.cuh"  // dispatch_sm60
#include "../pairwise_matrix/params.cuh"         // pairwise_matrix_params
#include <raft/util/arch.cuh>                    // raft::util::arch::SM_*

// NOTE: to minimize compile times, we do not include dispatch_sm80.cuh.
// Including dispatch_sm80.cuh can slow down compile times (due to CUTLASS).
// Therefore, it is the including file's responsibility to include the correct
// dispatch_smXX.cuh headers, as is done in cuvs/distance/detail/distance.cuh
// and src/distance/detail/pairwise_matrix/dispatch_*.cu.

namespace cuvs::distance::detail {

// This forward-declaration ensures that we do not need to include
// dispatch_sm80.cuh if we are not calling it in practice. This makes compiling
// all the non-CUTLASS based distance instantiations faster. For CUTLASS-based
// distances, dispatch_sm80.cuh has to be included by the file including this
// file.
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(OpT,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>,
                                   SM_compat_t,
                                   cudaStream_t);

#ifdef CUVS_CK_ENABLED
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_ck_dispatch(OpT,
                                 pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>,
                                 SM_compat_t,
                                 cudaStream_t);
#endif

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void pairwise_matrix_dispatch(OpT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const OutT* x_norm,
                              const OutT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major)
{
  // Create kernel parameter struct. Flip x and y if column major.
  IdxT ldx    = is_row_major ? k : m;
  IdxT ldy    = is_row_major ? k : n;
  IdxT ld_out = is_row_major ? n : m;

  pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params{
    m, n, k, ldx, ldy, ld_out, x, y, x_norm, y_norm, out, fin_op, is_row_major};

  if (!params.is_row_major) { params.flip_x_and_y(); }

  // Dispatch rule:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel below SM_80
  namespace arch = raft::util::arch;

#ifdef CUVS_CK_ENABLED
  if constexpr (ops::has_ck_op<OpT>()) {
    auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
    pairwise_matrix_ck_dispatch(distance_op, params, any_range, stream);
  } else {
    auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
    pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
  }
#elif !defined(__HIP_PLATFORM_AMD__)
  constexpr bool cutlass_op_unavailable = !ops::has_cutlass_op<OpT>();

  if constexpr (cutlass_op_unavailable) {
    // Always execute legacy kernels when no cutlass op is available
    auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
    pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
  } else {
    auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());
    auto legacy_range  = arch::SM_range(arch::SM_min(), arch::SM_80());

    // Get pointer to SM60 kernel to determine the best compute architecture
    // out of all for which the kernel was compiled for that matches closely
    // to the current device. Other methods to determine the architecture (that do not
    // require a pointer) can be error prone. See:
    // https://github.com/NVIDIA/cub/issues/545
    auto sm60_wrapper = pairwise_matrix_sm60_get_wrapper(distance_op, params, legacy_range);
    void* kernel_ptr  = reinterpret_cast<void*>(sm60_wrapper.kernel_ptr);
    auto runtime_arch = arch::kernel_virtual_arch(kernel_ptr);

    // TODO: the cutlass doesn't support the odd `k` on half DataT.
    bool if_unsupported_on_half = (sizeof(DataT) == 2) && ((k % 2) != 0);

    if (if_unsupported_on_half) {
      auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
      pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
    } else if (cutlass_range.contains(runtime_arch) && !if_unsupported_on_half) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      pairwise_matrix_sm80_dispatch(distance_op, params, cutlass_range, stream);
    } else {
      // Reuse kernel wrapper that we obtained above. This avoids performing the
      // dispatch twice.
      sm60_wrapper.launch(distance_op, params, stream);
    }
  }
#else
  // HIP without CK, or unknown platform: use the SM60 (legacy) path.
  auto any_range = arch::SM_range(arch::SM_min(), arch::SM_future());
  pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
#endif
}

};  // namespace cuvs::distance::detail
