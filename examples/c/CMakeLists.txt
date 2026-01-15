# =============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#
option(CUDA_BACKEND "Build for CUDA devices. Default configuration builds for AMD devices." OFF)

# CXX is needed for building dependencies
set(lang_list "C" "CXX")

if(CUDA_BACKEND)
  include(rapids-cuda)
  rapids_cuda_init_architectures(hipvs_c_examples)
  list(APPEND lang_list "CUDA")
else()
  include(rapids-hip)
  rapids_hip_init_architectures(hipvs_c_examples)
  list(APPEND lang_list "HIP")
endif()

project(hipvs_c_examples LANGUAGES ${lang_list})

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY ON)
include(../cmake/thirdparty/get_dlpack.cmake)
include(../cmake/thirdparty/get_cuvs.cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")

add_executable(CAGRA_C_EXAMPLE src/cagra_c_example.c)
target_include_directories(CAGRA_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(CAGRA_C_EXAMPLE PRIVATE cuvs::c_api $<TARGET_NAME_IF_EXISTS:conda_env>)

add_executable(L2_C_EXAMPLE src/L2_c_example.c)
target_include_directories(L2_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(L2_C_EXAMPLE PRIVATE cuvs::c_api $<TARGET_NAME_IF_EXISTS:conda_env>)

add_executable(IVF_FLAT_C_EXAMPLE src/ivf_flat_c_example.c)
target_include_directories(IVF_FLAT_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(IVF_FLAT_C_EXAMPLE PRIVATE cuvs::c_api $<TARGET_NAME_IF_EXISTS:conda_env>)

add_executable(IVF_PQ_C_EXAMPLE src/ivf_pq_c_example.c)
target_include_directories(IVF_PQ_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(IVF_PQ_C_EXAMPLE PRIVATE cuvs::c_api $<TARGET_NAME_IF_EXISTS:conda_env>)
