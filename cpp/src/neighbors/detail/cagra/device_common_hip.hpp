/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/warp_primitives.cuh>

namespace cuvs::neighbors::cagra::detail::device {

/**
 * @brief Get the shared memory base offset. Some values in your 64 bit flat address pointer fall
 * into the shared/LDS aperture.
 *
 */
RAFT_DEVICE_INLINE_FUNCTION uint64_t shared_base_offset()
{
  uint64_t addr{};
  asm volatile("s_mov_b64 %0, src_shared_base \n"  // src_shared_base is only available on GFX9+
               : "=s"(addr));
  return addr;
}
/**
 * @brief Convert a pointer in flat address space to an offset into the shared memory
 *
 * @tparam T Type of the value pointed to by ptr
 * @param ptr Pointer in flat/generic address space
 * @return RAFT_DEVICE_INLINE_FUNCTION
 */
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION uint32_t flat_to_shared_offset(const T* ptr)
{
  return static_cast<uint32_t>(reinterpret_cast<uint64_t>(ptr) - shared_base_offset());
}

/**
 * @brief Retrieve the flat address from LDS offset
 *
 * @tparam T Type of the value pointed to by the pointer returned by this function.
 * @param offset LDS offset computed with flat_to_shared_offset.
 * @return RAFT_DEVICE_INLINE_FUNCTION*
 */
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION T* shared_offset_to_flat(uint32_t offset)
{
  return reinterpret_cast<T*>(shared_base_offset() + offset);
}
}  // namespace cuvs::neighbors::cagra::detail::device
