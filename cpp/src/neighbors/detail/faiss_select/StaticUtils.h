/**
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include <hip/hip_runtime.h>
#include <raft/util/cuda_dev_essentials.cuh>

// allow usage for non-CUDA files
#ifndef __host__
#define __host__
#define __device__
#endif

namespace cuvs::neighbors::detail::faiss_select::utils {

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v)
{
  return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v)
{
  return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (raft::log2(v) + 1)));
}

static_assert(nextHighestPowerOf2(1) == 2, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(2) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(3) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(4) == 8, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(15) == 16, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(16) == 32, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(17) == 32, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(1536000000u) == 2147483648u, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2((size_t)2147483648ULL) == (size_t)4294967296ULL,
              "nextHighestPowerOf2");

}  // namespace cuvs::neighbors::detail::faiss_select::utils
