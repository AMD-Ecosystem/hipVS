/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hiprtc.h>
#else
#include <nvJitLink.h>
#endif

#include "nvjitlink_checker.hpp"

#ifdef __HIP_PLATFORM_AMD__
// GPU architecture families for which a distinct JIT-LTO bitcode fragment is
// pre-compiled at build time. At runtime the live device's gfx arch is mapped to
// one of these families and the matching fragment is selected before linking.
//
// Most families are LLVM gfxN-generic targets, which one bitcode can serve for
// every documented member of that family. gfx90a (MI200) is the exception: LLVM
// provides no generic target covering it (gfx9-generic is documented as
// gfx900/902/904/906/909/90c only), so that slot holds concrete gfx90a bitcode.
//
// The fragment must already target the live device, or a generic the device is a
// documented member of. The hiprtc link step is passed --offload-arch=<device>,
// but that only re-stamps the code object's EF_AMDGPU_MACH -- LLVM drives codegen
// from the per-function "target-cpu"/"target-features" attributes frozen into the
// bitcode at build time, so the link cannot retarget a mismatched fragment.
//
// The order of these enumerators is significant: it must match the order of the
// per-family data/length tables emitted by register_fatbin.cpp.in, and the
// _JIT_LTO_ALL_FAMILY_ARCHES list in generate_jit_lto_kernels.cmake.
enum class ArchFamily : int {
  Gfx90a = 0,  // gfx90a         (concrete; MI200)
  Gfx9_4,      // gfx9-4-generic (gfx942 MI300 / gfx950 MI350)
  Gfx11,       // gfx11-generic  (gfx11xx / gfx115x)
  Gfx12,       // gfx12-generic  (gfx12xx)
  Count
};

inline constexpr int kNumArchFamilies = static_cast<int>(ArchFamily::Count);

inline constexpr int arch_family_index(ArchFamily family) { return static_cast<int>(family); }
#endif

struct FragmentEntry {
  virtual ~FragmentEntry() = default;

#ifdef __HIP_PLATFORM_AMD__
  virtual bool add_to(hiprtcLinkState state) const = 0;
#else
  virtual bool add_to(nvJitLinkHandle& handle) const = 0;
#endif

  virtual const char* get_key() const = 0;

#ifdef __HIP_PLATFORM_AMD__
  // Select which pre-compiled per-family bitcode this fragment should expose via
  // get_data()/get_length(). No-op for fragments that are not family-specific
  // (e.g. runtime-compiled UDF fragments).
  virtual void set_arch_family(ArchFamily) {}
#endif
};

struct FatbinFragmentEntry : FragmentEntry {
  virtual const uint8_t* get_data() const = 0;

  virtual size_t get_length() const = 0;

#ifdef __HIP_PLATFORM_AMD__
  virtual hiprtcJITInputType get_input_type() const { return HIPRTC_JIT_INPUT_LLVM_BITCODE; }

  bool add_to(hiprtcLinkState state) const override final;
#else
  bool add_to(nvJitLinkHandle& handle) const override final;
#endif
};

template <typename FragmentTag>
struct StaticFatbinFragmentEntry final : FatbinFragmentEntry {
#ifdef __HIP_PLATFORM_AMD__
  const uint8_t* get_data() const override
  {
    const uint8_t* p = StaticFatbinFragmentEntry<FragmentTag>::data[checked_family_index()];
    if (p == nullptr) {
      throw std::runtime_error(
        std::string("No JIT-LTO bitcode fragment was built for the selected GPU architecture "
                    "family (fragment key: ") +
        get_key() + "). Rebuild libcuvs with this family enabled in CUVS_JIT_LTO_ARCH_FAMILIES.");
    }
    return p;
  }

  size_t get_length() const override
  {
    return StaticFatbinFragmentEntry<FragmentTag>::length[checked_family_index()];
  }

  void set_arch_family(ArchFamily family) override { selected_family_ = family; }

  const char* get_key() const override
  {
    return typeid(StaticFatbinFragmentEntry<FragmentTag>).name();
  }

  // One entry per ArchFamily (see enum order). Unbuilt families are nullptr / 0.
  static const uint8_t* const data[kNumArchFamilies];
  static const size_t length[kNumArchFamilies];

 private:
  // Fail loudly rather than silently linking some other family's bitcode if a caller ever reaches
  // get_data()/get_length() without going through set_arch_family() first.
  int checked_family_index() const
  {
    if (selected_family_ == ArchFamily::Count) {
      throw std::runtime_error(
        std::string("JIT-LTO fragment used before its GPU architecture family was selected "
                    "(fragment key: ") +
        get_key() + ").");
    }
    return arch_family_index(selected_family_);
  }

  // No default family: AlgorithmPlanner::build() must call set_arch_family() before linking.
  ArchFamily selected_family_ = ArchFamily::Count;
#else
  const uint8_t* get_data() const override { return StaticFatbinFragmentEntry<FragmentTag>::data; }

  size_t get_length() const override { return StaticFatbinFragmentEntry<FragmentTag>::length; }

  const char* get_key() const override
  {
    return typeid(StaticFatbinFragmentEntry<FragmentTag>).name();
  }

  static const uint8_t* const data;
  static const size_t length;
#endif
};

struct UDFFatbinFragment final : FatbinFragmentEntry {
  UDFFatbinFragment(std::string key, std::vector<uint8_t> bytes)
    : key_(std::move(key)), bytes_(std::move(bytes))
  {
  }

  const uint8_t* get_data() const override { return bytes_.data(); }

  size_t get_length() const override { return bytes_.size(); }

  const char* get_key() const override { return key_.c_str(); }

 private:
  std::string key_;
  std::vector<uint8_t> bytes_;
};
