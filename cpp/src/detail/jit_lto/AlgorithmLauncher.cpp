/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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
