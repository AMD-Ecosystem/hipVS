/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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

#include <cuvs/detail/jit_lto/NVRTCLTOFragmentCompiler.hpp>

#include <mutex>

#include <raft/core/error.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#include <cuvs/nvrtc.h>
#else
#include "cuda.h"
#include <nvrtc.h>
#endif

#define NVRTC_SAFE_CALL(_call)                                                 \
  {                                                                            \
    nvrtcResult result = _call;                                                \
    std::string error_string =                                                 \
      std::string("nvrtc error: ") + std::string(nvrtcGetErrorString(result)); \
    RAFT_EXPECTS(result == NVRTC_SUCCESS, "%s", error_string.c_str());         \
  }

NVRTCLTOFragmentCompiler::NVRTCLTOFragmentCompiler()
{
#ifdef __HIP_PLATFORM_AMD__
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));

  hipDeviceProp_t prop{};
  RAFT_CUDA_TRY(cudaGetDeviceProperties(&prop, device));

  std::string gcn_arch(prop.gcnArchName);
  auto colon_pos = gcn_arch.find(':');
  if (colon_pos != std::string::npos) { gcn_arch = gcn_arch.substr(0, colon_pos); }

  this->standard_compile_opts = {
    std::string{"--offload-arch=" + gcn_arch},
    std::string{"-fgpu-rdc"},
    std::string{"--std=c++20"},
    std::string{"-default-device"},
  };
#else
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  this->standard_compile_opts = {
    std::string{"-arch=sm_" + std::to_string((major * 10 + minor))},
    std::string{"-dlto"},
    std::string{"-rdc=true"},
    std::string{"--std=c++20"},
    std::string{"-default-device"},
  };
#endif
}

std::unique_ptr<UDFFatbinFragment> NVRTCLTOFragmentCompiler::read_cache(
  std::string const& key) const
{
  std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
  if (auto it = cache.find(key); it != cache.end()) {
    return std::make_unique<UDFFatbinFragment>(key, it->second);
  }
  return nullptr;
}

std::unique_ptr<UDFFatbinFragment> NVRTCLTOFragmentCompiler::compile(std::string const& key,
                                                                     std::string const& code)
{
  if (auto hit = read_cache(key)) { return hit; }

  std::unique_lock<std::shared_mutex> write_lock(cache_mutex_);
  if (auto it = cache.find(key); it != cache.end()) {
    return std::make_unique<UDFFatbinFragment>(key, it->second);
  }

  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog, code.c_str(), "nvrtc_lto_fragment", 0, nullptr, nullptr));

  // Convert std::vector<std::string> to std::vector<const char*> for nvrtc API
  std::vector<const char*> opts;
  opts.reserve(this->standard_compile_opts.size());
  for (const auto& opt : this->standard_compile_opts) {
    opts.push_back(opt.c_str());
  }

  nvrtcResult compileResult = nvrtcCompileProgram(prog,          // prog
                                                  opts.size(),   // numOptions
                                                  opts.data());  // options

  try {
    if (compileResult != NVRTC_SUCCESS) {
      // Obtain compilation log from the program.
      size_t log_size;
      NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
      std::unique_ptr<char[]> log{new char[log_size]};
      NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
      RAFT_FAIL("nvrtc compile error log: \n%s", log.get());
    }
  } catch (...) {
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    throw;
  }

  // Obtain generated LTO IR from the program.
  std::size_t ltoIRSize;
  NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoIRSize));

  std::vector<uint8_t> lto_ir(ltoIRSize);
  NVRTC_SAFE_CALL(nvrtcGetLTOIR(prog, reinterpret_cast<char*>(lto_ir.data())));

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  cache[key] = std::move(lto_ir);
  return std::make_unique<UDFFatbinFragment>(key, cache[key]);
}

NVRTCLTOFragmentCompiler& nvrtc_compiler()
{
  static NVRTCLTOFragmentCompiler compiler;
  return compiler;
}
