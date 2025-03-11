# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

# We always need CUDA for cuvs because the cuvs dependency brings in a header-only cuco dependency
# that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cuvs_py)

project(
  cuvs_py
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

# ##################################################################################################
# * Process User Options  ------------------------------------------------------

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cython-core)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

# --- cuVS ---#
find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)

# --- dlpack ---#
include(../../cpp/cmake/thirdparty/get_dlpack.cmake)

# ensure Cython targets can find dlpack headers (these do not come installed with with cuVS)
target_include_directories(cuvs::cuvs INTERFACE "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")

# ##################################################################################################
# * Build Cython artifacts -----------------------------------------------------

rapids_cython_init()

add_subdirectory(cuvs/common)
add_subdirectory(cuvs/distance)
add_subdirectory(cuvs/neighbors)
add_subdirectory(cuvs/preprocessing)
