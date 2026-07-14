/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/nvjitlink_checker.hpp>

#include <memory>
#include <string>

#include <raft/core/error.hpp>

#ifdef __HIP_PLATFORM_AMD__

void check_hiprtc_link_result(hiprtcResult result)
{
  if (result != HIPRTC_SUCCESS) {
    std::string error_msg = "hipRTC link error: ";
    error_msg += hiprtcGetErrorString(result);
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
