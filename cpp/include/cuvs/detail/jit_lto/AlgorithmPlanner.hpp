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

#include <memory>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AlgorithmLauncher.hpp"
#include "FragmentEntry.hpp"

struct LauncherJitCache {
  std::shared_mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
};

struct AlgorithmPlanner {
  AlgorithmPlanner(std::string entrypoint, LauncherJitCache& jit_cache)
    : entrypoint(std::move(entrypoint)), jit_cache_(jit_cache)
  {
  }

  std::shared_ptr<AlgorithmLauncher> get_launcher();

  std::string entrypoint;
  std::vector<std::unique_ptr<FragmentEntry>> fragments;

  template <typename T, typename = std::enable_if_t<std::is_convertible_v<T*, FragmentEntry*>>>
  void add_fragment(std::unique_ptr<T> fragment)
  {
    fragments.push_back(std::unique_ptr<FragmentEntry>(std::move(fragment)));
  }

  template <typename FragmentTag>
  void add_static_fragment()
  {
    add_fragment(std::make_unique<StaticFatbinFragmentEntry<FragmentTag>>());
  }

 protected:
  /** Extra link-time option strings passed to hiprtcLink via hipJitOptionIRtoISAOptExt.
   *  Base build() always passes "--offload-arch=gfxXXX" first; derived planners may
   *  append here in their constructor body. */
  std::vector<std::string> linktime_extra_options;

 private:
  std::string get_fragments_key() const;
  std::shared_ptr<AlgorithmLauncher> build();

  std::shared_ptr<AlgorithmLauncher> read_cache(std::string const& launch_key) const;

  LauncherJitCache& jit_cache_;
};
