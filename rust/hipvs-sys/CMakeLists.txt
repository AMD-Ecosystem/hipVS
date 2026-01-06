# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
# MIT License
#
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

# we want to use the already built libcuvs if its available, but the rust cmake-rs project doesn't
# support anything like find_package https://github.com/rust-lang/cmake-rs/issues/111 instead we're
# adding an extra level of indirection here - cmake-rs will attempt to build this project, and we'll
# using the existing libcuvs if its already built, and only fall back to building libcuvs if it
# isn't

project(
  cuvs-rs
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX HIP
)

option(FIND_CUVS_CPP "Search for existing CUVS C++ installations before defaulting to local files"
       ON
)

# If the user requested it we attempt to find CUVS.
if(FIND_CUVS_CPP)
  find_package(cuvs REQUIRED COMPONENTS compiled c_api)
  if(NOT TARGET cuvs::cuvs)
    message(
      FATAL_ERROR
        "Building against a preexisting libcuvs library requires the compiled libcuvs to have been built!"
    )
  endif()
else()
  set(cuvs_FOUND OFF)
endif()

if(NOT cuvs_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY ON)
  add_subdirectory(../../cpp cuvs-cpp EXCLUDE_FROM_ALL)
endif()

get_target_property(_cuvs_c_api_path cuvs::c_api IMPORTED_LOCATION_RELEASE)
if(_cuvs_c_api_path)
  # CUVS_C_LIBRARY_SO_PATH is later used in build.rs to link against the C library
  set(CUVS_C_LIBRARY_SO_PATH
      "${_cuvs_c_api_path}"
      CACHE FILEPATH "Full path to libcuvs_c.so (imported cuvs::c_api)" FORCE
  )
endif()

include(get_dlpack.cmake)

# add a dummy target here,
add_library(cuvs-rust INTERFACE)
target_link_libraries(cuvs-rust INTERFACE cuvs::cuvs)
install(TARGETS cuvs-rust)
