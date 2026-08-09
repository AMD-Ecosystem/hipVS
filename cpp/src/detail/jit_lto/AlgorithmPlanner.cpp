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

#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/nvjitlink_checker.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <cstring>
#include <cuvs/cuda_runtime.h>
#include <hip/hiprtc.h>
#else
#include "cuda_runtime.h"
#include "nvJitLink.h"
#endif

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

std::string AlgorithmPlanner::get_fragments_key() const
{
  std::string key = "";
#ifdef __HIP_PLATFORM_AMD__
  // On AMD, hipModuleLoadData loads the JIT-linked code object into the *current* device's
  // context. The static cache is process-wide, so when multiple OMP threads call get_launcher()
  // on different devices concurrently the first thread to build populates the cache with a module
  // handle that is only valid on its device. Subsequent threads on other devices get the same
  // cached hipFunction_t and fail with hipErrorInvalidDeviceFunction. Prefix the key with the
  // device ordinal so every device gets its own compiled and loaded module.
  // TODO(HIP): This is a HIP limitation — hipModule_t handles are device-context-specific, unlike
  // NVIDIA's cudaLibrary_t which is device-independent. A cleaner long-term fix would be to either
  // use hipModuleLoadDataEx with the multi-device context API (if/when ROCm exposes one), or
  // restructure the cache to be per-device rather than patching the key here.
  int device = 0;
  (void)cudaGetDevice(&device);
  key += "dev" + std::to_string(device) + ":";
#endif
  for (const auto& fragment : this->fragments) {
    key += fragment->get_key();
  }
  return key;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::read_cache(std::string const& launch_key) const
{
  auto& launchers = jit_cache_.launchers;
  std::shared_lock<std::shared_mutex> read_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }
  return nullptr;
}

#ifdef __HIP_PLATFORM_AMD__
namespace {

// Map a concrete gfx architecture name (colon-stripped, e.g. "gfx1151") to the
// pre-compiled bitcode family. Throws if the device is not covered by any family
// that this library was built with. Keep in sync with the CUVS_JIT_LTO_ARCH_FAMILIES
// list in cpp/CMakeLists.txt and the ArchFamily enum in FragmentEntry.hpp.
ArchFamily arch_family_from_gcn_arch(std::string const& gcn_arch)
{
  const auto starts_with = [&](const char* prefix) { return gcn_arch.rfind(prefix, 0) == 0; };

  // gfx9-4 family (CDNA3/CDNA "MI300"-class): must be checked before generic gfx9.
  if (starts_with("gfx94") || starts_with("gfx950")) { return ArchFamily::Gfx9_4; }
  // gfx9 family (Vega / CDNA1 / CDNA2).
  if (starts_with("gfx9")) { return ArchFamily::Gfx9; }
  // gfx11 family (RDNA3 / RDNA3.5).
  if (starts_with("gfx11")) { return ArchFamily::Gfx11; }
  // gfx12 family (RDNA4).
  if (starts_with("gfx12")) { return ArchFamily::Gfx12; }

  RAFT_FAIL(
    "No JIT-LTO bitcode fragment family is available for GPU architecture '%s'. This build of "
    "libcuvs only supports the gfx9, gfx9-4, gfx11 and gfx12 families. Rebuild with the required "
    "family added to CUVS_JIT_LTO_ARCH_FAMILIES.",
    gcn_arch.c_str());
}

}  // namespace
#endif

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::get_launcher()
{
  auto& launchers = jit_cache_.launchers;
  auto launch_key = this->get_fragments_key();

  if (auto hit = read_cache(launch_key)) { return hit; }

  std::unique_lock<std::shared_mutex> write_lock(jit_cache_.mutex);
  if (auto it = launchers.find(launch_key); it != launchers.end()) { return it->second; }

  std::string log_message =
    "JIT compiling launcher for kernel: " + this->entrypoint + " and device functions: ";
  for (const auto& fragment : this->fragments) {
    log_message += std::string{fragment->get_key()} + ",";
  }
  log_message.pop_back();
  RAFT_LOG_DEBUG("%s", log_message.c_str());
  auto launcher         = this->build();
  launchers[launch_key] = launcher;
  return launcher;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::build()
{
#ifdef __HIP_PLATFORM_AMD__
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));

  hipDeviceProp_t prop{};
  RAFT_CUDA_TRY(cudaGetDeviceProperties(&prop, device));

  std::string gcn_arch(prop.gcnArchName);
  const auto colon_pos = gcn_arch.find(':');
  if (colon_pos != std::string::npos) { gcn_arch = gcn_arch.substr(0, colon_pos); }

  // Select the pre-compiled bitcode fragment matching this device's family. The
  // hiprtc link below then specializes that (generic) bitcode to the exact arch.
  const ArchFamily family = arch_family_from_gcn_arch(gcn_arch);
  for (const auto& frag : this->fragments) {
    frag->set_arch_family(family);
  }

  const std::string arch_opt = "--offload-arch=" + gcn_arch;

  std::vector<const char*> ir_opts;
  ir_opts.push_back(arch_opt.c_str());
  for (auto const& opt : linktime_extra_options) {
    ir_opts.push_back(opt.c_str());
  }

  const size_t ir_opt_count  = ir_opts.size();
  hiprtcJIT_option options[] = {hipJitOptionIRtoISAOptExt, hipJitOptionIRtoISAOptCountExt};
  void* option_values[]      = {static_cast<void*>(ir_opts.data()),
                                reinterpret_cast<void*>(ir_opt_count)};

  hiprtcLinkState link_state;
  auto result = hiprtcLinkCreate(2, options, option_values, &link_state);
  check_hiprtc_link_result(result);

  for (const auto& frag : this->fragments) {
    frag->add_to(link_state);
  }

  void* bin_out   = nullptr;
  size_t bin_size = 0;
  result          = hiprtcLinkComplete(link_state, &bin_out, &bin_size);
  check_hiprtc_link_result(result);

  // bin_out is only valid until hiprtcLinkDestroy — copy it
  const std::vector<char> code_object(static_cast<char*>(bin_out),
                                      static_cast<char*>(bin_out) + bin_size);

  result = hiprtcLinkDestroy(link_state);
  RAFT_EXPECTS(result == HIPRTC_SUCCESS, "hiprtcLinkDestroy failed");

  cudaLibrary_t library;
  RAFT_CUDA_TRY(hipModuleLoadData(&library, code_object.data()));

  cudaKernel_t kernel;
  RAFT_CUDA_TRY(hipModuleGetFunction(&kernel, library, this->entrypoint.c_str()));

  return std::make_shared<AlgorithmLauncher>(kernel, library);

#else
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  std::vector<const char*> lopts;
  lopts.reserve(2 + linktime_extra_options.size());
  lopts.push_back("-lto");
  lopts.push_back(archs.c_str());
  for (auto const& opt : linktime_extra_options) {
    lopts.push_back(opt.c_str());
  }
  auto result = nvJitLinkCreate(&handle, static_cast<unsigned int>(lopts.size()), lopts.data());
  check_nvjitlink_result(handle, result);

  for (const auto& frag : this->fragments) {
    frag->add_to(handle);
  }

  // Call to nvJitLinkComplete causes linker to link together all the LTO-IR
  // modules perform any optimizations and generate cubin from it.
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  // get cubin from nvJitLink
  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  RAFT_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  // cubin is linked, so now load it
  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  cudaKernel_t kernel;
  RAFT_CUDA_TRY(cudaLibraryGetKernel(&kernel, library, this->entrypoint.c_str()));

  return std::make_shared<AlgorithmLauncher>(kernel, library);
#endif
}
