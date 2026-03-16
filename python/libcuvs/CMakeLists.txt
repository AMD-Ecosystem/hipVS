# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
# Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc. Permission is hereby granted,
# free of charge, to any person obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject to the
# following conditions: The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
# WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ===============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/../../overrides.cmake")

set(lang_list "CXX" "HIP")

project(
  libcuvs-python
  VERSION ${RAPIDS_VERSION}
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
# <<<<<<< OLD CODE from d07a3b6b2 (04f402a00) - COMMENTED OUT
# set(CUDA_STATIC_RUNTIME OFF) # No static linking with HIP, HIP must be installed
# set(CUDA_STATIC_MATH_LIBRARIES OFF) # No static linking with HIP, HIP must be installed
# =======
set(CUDA_STATIC_MATH_LIBRARIES OFF)
# >>>>>>> END NEW CODE (04f402a00)

# --- RAFT ---#
set(CUVS_STATIC_RAPIDS_LIBRARIES OFF)

# --- cuVS ---#
set(BUILD_TESTS OFF)
set(BUILD_C_LIBRARY ON)
set(CUVS_COMPILE_DYNAMIC_ONLY ON)

add_subdirectory(../../cpp cuvs-cpp)

set(rpaths
    "$ORIGIN/../../../../" # Assumes the following conda structure:
                           # $ORIGIN=$CONDA_PREFIX/lib/site-packages/libcuvs/lib/
    "/opt/rocm/lib/llvm/lib/" # Assumes ROCm is installed in /opt/rocm. This is for libomp.so
)

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
