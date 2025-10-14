# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Use RAPIDS_VERSION_MAJOR_MINOR from rapids_config.cmake
# TODO(AMD/HIP): Update the default values
set(CUVS_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(CUVS_FORK "AMD-AIOSS")
#set(CUVS_PINNED_TAG "branch-${RAPIDS_VERSION_MAJOR_MINOR}")
set(CUVS_PINNED_TAG "release/rocmds-25.10")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG ENABLE_NVTX CLONE_ON_PIN BUILD_CPU_ONLY BUILD_SHARED_LIBS)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUVS_VERSION}")
        message(STATUS "cuVS: pinned tag found: ${PKG_PINNED_TAG}. Cloning cuVS locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    endif()

    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-bench-exports
            INSTALL_EXPORT_SET  cuvs-bench-exports
            COMPONENTS          cuvs
            CPM_ARGS
              GIT_REPOSITORY        https://github.com/${PKG_FORK}/cuvs.git
              GIT_TAG               ${PKG_PINNED_TAG}
              SOURCE_SUBDIR         cpp
              OPTIONS
              "BUILD_SHARED_LIBS ${PKG_BUILD_SHARED_LIBS}"
              "BUILD_CPU_ONLY ${PKG_BUILD_CPU_ONLY}"
              "BUILD_TESTS OFF"
              "BUILD_CAGRA_HNSWLIB OFF"
              "CUVS_CLONE_ON_PIN ${PKG_CLONE_ON_PIN}"
            )
endfunction()


# Change pinned tag here to test a commit in CI
# To use a different cuVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}.00
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        ENABLE_NVTX              OFF
        # When PINNED_TAG above doesn't match the default rapids branch,
        # force local cuvs clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ${CUVS_CLONE_ON_PIN}
        BUILD_CPU_ONLY ${BUILD_CPU_ONLY}
        BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}
)
