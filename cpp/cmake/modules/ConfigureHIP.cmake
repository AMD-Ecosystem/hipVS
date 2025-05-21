# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND CUVS_CXX_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
  list(APPEND CUVS_GPU_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
endif()

# Be very strict when compiling with GCC as host compiler (and thus more lenient when compiling with
# clang)
if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND CUVS_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
  list(APPEND CUVS_GPU_FLAGS -Wall,-Werror,-Wno-error=deprecated-declarations)

  # set warnings as errors if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2.0)
  list(APPEND CUVS_GPU_FLAGS -Werror=all-warnings)
  # endif()
endif()
# Allow invalid CUDA kernels in the short term
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
  list(APPEND CUVS_GPU_FLAGS -static-global-template-stub=false)
endif()

# TODO(HIP/AMD): port this if(CUDA_LOG_COMPILE_TIME) list(APPEND CUVS_GPU_FLAGS
# "--time=nvcc_compile_log.csv") endif()

# list(APPEND CUVS_GPU_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUVS_CXX_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
list(APPEND CUVS_GPU_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
# make sure we produce smallest binary size list(APPEND CUVS_GPU_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  message(FATAL_ERROR "CUVS does not support line-number information for hip platform")
endif()

if(OpenMP_FOUND)
  list(APPEND CUVS_GPU_FLAGS ${OpenMP_CXX_FLAGS})
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUVS: Building with debugging flags")
  list(APPEND CUVS_GPU_FLAGS -g)
endif()
