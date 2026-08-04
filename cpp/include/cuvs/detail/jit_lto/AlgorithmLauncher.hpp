/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cstdint>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>
#endif
#include <string>
#include <unordered_map>

#include <memory>

struct AlgorithmLauncher {
  AlgorithmLauncher() : kernel{nullptr}, library{nullptr} {}

  AlgorithmLauncher(cudaKernel_t k, cudaLibrary_t lib);

  ~AlgorithmLauncher();

  AlgorithmLauncher(const AlgorithmLauncher&)            = delete;
  AlgorithmLauncher& operator=(const AlgorithmLauncher&) = delete;

  AlgorithmLauncher(AlgorithmLauncher&& other) noexcept;
  AlgorithmLauncher& operator=(AlgorithmLauncher&& other) noexcept;

  template <typename FuncT, typename... Args>
  void dispatch(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    static_assert(std::is_same_v<FuncT, void(std::remove_reference_t<Args>...)>,
                  "dispatch() argument types do not match the kernel function signature FuncT");

    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  template <typename FuncT, typename... Args>
  void dispatch_cooperative(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    static_assert(
      std::is_same_v<FuncT, void(std::remove_reference_t<Args>...)>,
      "dispatch_cooperative() argument types do not match the kernel function signature FuncT");

    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call_cooperative(stream, grid, block, shared_mem, kernel_args);
  }

  cudaKernel_t get_kernel() { return this->kernel; }

 private:
  void call(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  void call_cooperative(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  cudaKernel_t kernel;
  cudaLibrary_t library;
};
