/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T, typename AccT, int Veclen>
__device__ void compute_dist_euclidean_impl(AccT& acc, AccT x, AccT y)
{
  if constexpr (std::is_same_v<T, uint8_t> && std::is_same_v<AccT, uint32_t>) {
    if constexpr (Veclen > 1) {
      const auto diff = [=]() {
#ifdef __HIP_PLATFORM_AMD__
        static_assert(sizeof(uchar4) == sizeof(uint32_t));
        uchar4 ux = *reinterpret_cast<const uchar4*>(&x);
        uchar4 uy = *reinterpret_cast<const uchar4*>(&y);

        uchar4 diff;
        diff.x = __builtin_abs(ux.x - uy.x);
        diff.y = __builtin_abs(ux.y - uy.y);
        diff.z = __builtin_abs(ux.z - uy.z);
        diff.w = __builtin_abs(ux.w - uy.w);

        return *reinterpret_cast<uint32_t*>(&diff);
#else
        return __vabsdiffu4(x, y);
#endif
      }();
      acc = raft::dp4a(diff, diff, acc);
    } else {
#ifdef __HIP_PLATFORM_AMD__
      const int diff = (x > y) ? (x - y) : (y - x);
#else
      const auto diff = __usad(x, y, 0u);
#endif
      acc += diff * diff;
    }
  } else if constexpr (std::is_same_v<T, int8_t> && std::is_same_v<AccT, int32_t>) {
    if constexpr (Veclen > 1) {
      const auto diff = [=]() {
#ifdef __HIP_PLATFORM_AMD__
        static_assert(sizeof(char4) == sizeof(int32_t));
        char4 sx = *reinterpret_cast<const char4*>(&x);
        char4 sy = *reinterpret_cast<const char4*>(&y);

        uchar4 diff;
        diff.x = abs(sx.x - sy.x);
        diff.y = abs(sx.y - sy.y);
        diff.z = abs(sx.z - sy.z);
        diff.w = abs(sx.w - sy.w);

        return *reinterpret_cast<uint32_t*>(&diff);
#else
        return __vabsdiffs4(x, y);
#endif
      }();
      acc = raft::dp4a(diff, diff, static_cast<uint32_t>(acc));
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  } else {
    const auto diff = x - y;
    acc += diff * diff;
  }
}

template <typename T, typename AccT, int Veclen>
__device__ void compute_dist_inner_product_impl(AccT& acc, AccT x, AccT y)
{
  if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
    acc = raft::dp4a(x, y, acc);
  } else {
    acc += x * y;
  }
}

}  // namespace cuvs::neighbors::ivf_flat::detail
