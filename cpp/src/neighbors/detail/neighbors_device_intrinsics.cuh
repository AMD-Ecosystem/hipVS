/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/warp_primitives.cuh>
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#include <cfloat>
#include <cstdint>

namespace cuvs::neighbors::detail::device {

// warpSize for compile time calculation
constexpr unsigned warp_size = raft::WarpSize;

// using LOAD_256BIT_T = ulonglong4;
using LOAD_128BIT_T = uint4;
using LOAD_64BIT_T  = uint64_t;

template <class LOAD_T, class DATA_T>
RAFT_DEVICE_INLINE_FUNCTION constexpr unsigned get_vlen()
{
  static_assert(sizeof(DATA_T) > 0, "get_vlen: DATA_T must have positive size");
  return static_cast<unsigned>(sizeof(LOAD_T) / sizeof(DATA_T));
}

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

template <uint32_t TeamSize, typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x) -> T
{
#pragma unroll
  for (uint32_t stride = TeamSize >> 1; stride > 0; stride >>= 1) {
    x += __shfl_xor_sync(__activemask(), x, stride, TeamSize);
  }
  return x;
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x, uint32_t team_size_bitshift) -> T
{
  // HIP/AMD: raft::shfl_xor breaks single-CTA search convergence when only a subset of
  // wavefront lanes participate; use the intrinsic directly (see hipVS issue #20).
  const uint32_t team_width = raft::warp_size();
  switch (team_size_bitshift) {
    case 6: x += __shfl_xor_sync(__activemask(), x, 32, team_width); [[fallthrough]];
    case 5: x += __shfl_xor_sync(__activemask(), x, 16, team_width); [[fallthrough]];
    case 4: x += __shfl_xor_sync(__activemask(), x, 8, team_width); [[fallthrough]];
    case 3: x += __shfl_xor_sync(__activemask(), x, 4, team_width); [[fallthrough]];
    case 2: x += __shfl_xor_sync(__activemask(), x, 2, team_width); [[fallthrough]];
    case 1: x += __shfl_xor_sync(__activemask(), x, 1, team_width); [[fallthrough]];
    default: return x;
  }
}

template <uint32_t Dim = 1024, uint32_t Stride = 128, typename T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto swizzling(T x) -> T
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  if constexpr (Stride <= 32) {
    return x;
  } else if constexpr (Dim <= 1024) {
    return x ^ (x >> 5);
  } else {
    return x ^ ((x >> 5) & 0x1f);
  }
}

}  // namespace cuvs::neighbors::detail::device

// CAGRA JIT kernels extend `cuvs::neighbors::cagra::detail::device` in other headers; re-export
// the shared helpers there under the historical nested name.
namespace cuvs::neighbors::cagra::detail::device {
using namespace cuvs::neighbors::detail::device;
}  // namespace cuvs::neighbors::cagra::detail::device
