# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
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
#
# =============================================================================

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND CUVS_CXX_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
  list(APPEND CUVS_GPU_FLAGS -Xcompiler=-Wno-deprecated-declarations
       -DRAFT_HIDE_DEPRECATION_WARNINGS
  )
endif()

if(DISABLE_OPENMP)
  list(APPEND CUVS_CXX_FLAGS -Wno-unknown-pragmas)
  list(APPEND CUVS_CUDA_FLAGS -Xcompiler=-Wno-unknown-pragmas)
endif()

# Be very strict when compiling with GCC as host compiler (and thus more lenient when compiling with
# clang)
if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND CUVS_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations
       -Wno-reorder
  )
  list(APPEND CUVS_GPU_FLAGS
       -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations,-Wno-reorder
  )

  # set warnings as errors
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2.0)
    list(APPEND CUVS_GPU_FLAGS -Werror=all-warnings)
  endif()
endif()

# <<<<<<< OLD CODE from 2aca99707 (cffd0318c) - COMMENTED OUT
# # Allow invalid CUDA kernels in the short term
# if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
#   list(APPEND CUVS_GPU_FLAGS -static-global-template-stub=false)
# endif()
#
# =======
# >>>>>>> END NEW CODE (cffd0318c)
if(CUDA_LOG_COMPILE_TIME)
  list(APPEND CUVS_GPU_FLAGS "--time=nvcc_compile_log.csv")
endif()

list(APPEND CUVS_GPU_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUVS_CXX_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
list(APPEND CUVS_GPU_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
# make sure we produce smallest binary size
if(CUDA_BACKEND)
  include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
  rapids_cuda_enable_fatbin_compression(VARIABLE CUVS_CUDA_FLAGS TUNE_FOR rapids)
else()
  list(APPEND CUVS_GPU_FLAGS -Xfatbin=-compress-all)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
     AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9 AND CMAKE_CUDA_COMPILER_VERSION
                                                                     VERSION_LESS 13.0)
  )
    list(APPEND CUVS_GPU_FLAGS -Xfatbin=--compress-level=3)
  endif()
endif()

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  list(APPEND CUVS_GPU_FLAGS -lineinfo)
endif()

if(OpenMP_FOUND)
  list(APPEND CUVS_GPU_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "cuVS: Building with debugging flags")
  list(APPEND CUVS_GPU_FLAGS -G -Xcompiler=-rdynamic --maxrregcount=64)
  list(APPEND CUVS_GPU_FLAGS -Xptxas --suppress-stack-size-warning)
endif()
