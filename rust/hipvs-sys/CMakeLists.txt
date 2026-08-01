# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
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
# <<<<<<< OLD CODE from 76eb5a2b2 (fdf3862a0) - COMMENTED OUT
#   find_package(cuvs REQUIRED COMPONENTS compiled c_api)
#   if(NOT TARGET cuvs::cuvs)
#     message(
#       FATAL_ERROR
#         "Building against a preexisting libcuvs library requires the compiled libcuvs to have been built!"
#     )
#   endif()
# else()
#   set(cuvs_FOUND OFF)
# =======
  find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)
# >>>>>>> END NEW CODE (fdf3862a0)
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

# We are going to copy all the cuvs_c and dlpack headers into a staging location
if(TARGET cuvs_c)
  get_target_property(cuvs_c_headers cuvs_c INCLUDE_DIRECTORIES)
else()
  get_target_property(cuvs_c_headers cuvs::c_api INTERFACE_INCLUDE_DIRECTORIES)
endif()

file(
  INSTALL "${DLPACK_INCLUDE_DIR}"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bindings/"
  FILES_MATCHING
  PATTERN "*.h"
)

foreach(cuvs_c_dir IN LISTS cuvs_c_headers)
  if(EXISTS "${cuvs_c_dir}/cuvs/")
    file(
      INSTALL "${cuvs_c_dir}/cuvs"
      DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bindings/include/"
      FILES_MATCHING
      PATTERN "*.h"
    )
  endif()
endforeach()

# add a dummy target here,
add_library(cuvs-rust INTERFACE)
target_link_libraries(cuvs-rust INTERFACE cuvs::cuvs)
install(TARGETS cuvs-rust)
