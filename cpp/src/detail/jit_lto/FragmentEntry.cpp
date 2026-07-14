/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

#ifdef __HIP_PLATFORM_AMD__

bool FatbinFragmentEntry::add_to(hiprtcLinkState state) const
{
  auto result = hiprtcLinkAddData(state,
                                  get_input_type(),
                                  const_cast<void*>(static_cast<const void*>(get_data())),
                                  get_length(),
                                  get_key(),
                                  0,
                                  nullptr,
                                  nullptr);

  check_hiprtc_link_result(result);
  return true;
}

#else

bool FatbinFragmentEntry::add_to(nvJitLinkHandle& handle) const
{
  auto result = nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, get_data(), get_length(), get_key());

  check_nvjitlink_result(handle, result);
  return true;
}

#endif
