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

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

AlgorithmLauncher::AlgorithmLauncher(cudaKernel_t k, cudaLibrary_t lib) : kernel{k}, library{lib} {}

AlgorithmLauncher::~AlgorithmLauncher()
{
#ifdef __HIP_PLATFORM_AMD__
  if (library != nullptr) { (void)hipModuleUnload(library); }
#else
  if (library != nullptr) { (void)cudaLibraryUnload(library); }
#endif
}

AlgorithmLauncher::AlgorithmLauncher(AlgorithmLauncher&& other) noexcept
  : kernel{other.kernel}, library{other.library}
{
  other.kernel  = nullptr;
  other.library = nullptr;
}

AlgorithmLauncher& AlgorithmLauncher::operator=(AlgorithmLauncher&& other) noexcept
{
  if (this != &other) {
#ifdef __HIP_PLATFORM_AMD__
    if (library != nullptr) { (void)hipModuleUnload(library); }
#else
    if (library != nullptr) { cudaLibraryUnload(library); }
#endif
    kernel        = other.kernel;
    library       = other.library;
    other.kernel  = nullptr;
    other.library = nullptr;
  }
  return *this;
}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
#ifdef __HIP_PLATFORM_AMD__
  RAFT_CUDA_TRY(hipModuleLaunchKernel(kernel,
                                      grid.x,
                                      grid.y,
                                      grid.z,
                                      block.x,
                                      block.y,
                                      block.z,
                                      static_cast<unsigned int>(shared_mem),
                                      stream,
                                      kernel_args,
                                      nullptr));
#else
  cudaLaunchConfig_t config{};
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.dynamicSmemBytes = shared_mem;
  config.numAttrs         = 0;
  config.attrs            = NULL;

  RAFT_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
#endif
}

void AlgorithmLauncher::call_cooperative(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
#ifdef __HIP_PLATFORM_AMD__
  RAFT_CUDA_TRY(hipModuleLaunchCooperativeKernel(kernel,
                                                 grid.x,
                                                 grid.y,
                                                 grid.z,
                                                 block.x,
                                                 block.y,
                                                 block.z,
                                                 static_cast<unsigned int>(shared_mem),
                                                 stream,
                                                 kernel_args));
#else
  cudaLaunchAttribute attribute[1];
  attribute[0].id              = cudaLaunchAttributeCooperative;
  attribute[0].val.cooperative = 1;

  cudaLaunchConfig_t config{};
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.dynamicSmemBytes = shared_mem;
  config.numAttrs         = 1;
  config.attrs            = attribute;

  RAFT_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
#endif
}
