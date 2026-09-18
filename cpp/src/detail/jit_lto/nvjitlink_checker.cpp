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

#include <cuvs/detail/jit_lto/nvjitlink_checker.hpp>

#include <memory>
#include <string>

#include <raft/core/error.hpp>

#ifdef __HIP_PLATFORM_AMD__

void check_hiprtc_link_result(hiprtcResult result, const char* fragment_name, size_t fragment_size)
{
  if (result != HIPRTC_SUCCESS) {
    std::string error_msg = "hipRTC link error: ";
    error_msg += hiprtcGetErrorString(result);
    if (fragment_name != nullptr) {
      error_msg += " (while adding fragment '" + std::string{fragment_name} + "', " +
                   std::to_string(fragment_size) + " bytes)";
    }
    // hipRTC exposes no linker log: CLR accepts hipJitOptionErrorLogBuffer but never writes
    // to it (the hip headers mark it "CUDA Only"), and comgr's build log is only emitted to
    // the runtime's own logger. AMD_LOG_LEVEL=4 is the way to see why a fragment was rejected.
    error_msg += ". Re-run with AMD_LOG_LEVEL=4 for the runtime's linker diagnostics.";
    RAFT_FAIL("AlgorithmPlanner hipRTC error: %s", error_msg.c_str());
  }
}

#else

#include <nvJitLink.h>

void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS) {
    std::string error_msg = "nvJITLink failed with error " + std::to_string(result);
    size_t log_size       = 0;
    result                = nvJitLinkGetErrorLogSize(handle, &log_size);
    if (result == NVJITLINK_SUCCESS && log_size > 0) {
      std::unique_ptr<char[]> log{new char[log_size]};
      result = nvJitLinkGetErrorLog(handle, log.get());
      if (result == NVJITLINK_SUCCESS) { error_msg += "\n" + std::string(log.get()); }
    }
    RAFT_FAIL("AlgorithmPlanner nvJITLink error log: %s", error_msg.c_str());
  }
}

#endif
