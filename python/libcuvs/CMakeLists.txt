# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. Permission is hereby granted, free
# of charge, to any person obtaining a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions: The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ===============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/../../overrides.cmake")

set(lang_list "CXX")
if(CUDA_BACKEND)
  list(APPEND lang_list "CUDA")
else()
  list(APPEND lang_list "HIP")
endif()

project(
  libcuvs-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${lang_list}
)

option(USE_NCCL_RUNTIME_WHEEL "Use the NCCL wheel at runtime instead of the system library" OFF)

# Check if cuVS is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuvs package.
# cmake-format: off
# find_package(cuvs "${RAPIDS_VERSION}") Note: On the AMD/HIP platform we want to force libcuvs.so and libcuvs_c.so to be built and packaged into the wheel
# cmake-format: on

if(cuvs_FOUND)
  return()
endif()

unset(cuvs_FOUND)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME OFF) # No static linking with HIP, HIP must be installed
set(CUDA_STATIC_MATH_LIBRARIES OFF) # No static linking with HIP, HIP must be installed

# --- RAFT ---#
set(CUVS_USE_RAFT_STATIC OFF)

# --- cuVS ---#
set(BUILD_TESTS OFF)
set(BUILD_C_LIBRARY ON)
set(CUVS_COMPILE_DYNAMIC_ONLY ON)

add_subdirectory(../../cpp cuvs-cpp)

if(CUDA_BACKEND)
  # assumes libcuvs.so is installed 2 levels deep, e.g. site-packages/libcuvs/lib64/libcuvs.so
  set(rpaths
      "$ORIGIN/../../nvidia/cublas/lib"
      "$ORIGIN/../../nvidia/curand/lib"
      "$ORIGIN/../../nvidia/cusolver/lib"
      "$ORIGIN/../../nvidia/cusparse/lib"
      "$ORIGIN/../../nvidia/nvjitlink/lib"
  )

  if(USE_NCCL_RUNTIME_WHEEL)
    list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")
  endif()
else()
  set(rpaths
      "$ORIGIN/../../../../" # Assumes the following conda structure:
                             # $ORIGIN=$CONDA_PREFIX/lib/site-packages/libcuvs/lib/
      "/opt/rocm/lib/llvm/lib/" # Assumes ROCm is installed in /opt/rocm. This is for libomp.so
  )
endif()

set_property(
  TARGET cuvs
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cuvs_c
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
