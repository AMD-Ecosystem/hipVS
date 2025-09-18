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

set(CUVS_VERSION "0.1.0")
set(CUVS_FORK "AMD-AIOSS")
set(CUVS_PINNED_TAG "amd-integration")
# When PINNED_TAG above doesn't match the default branch,
# force local hipVS clone in build directory
# even if it's already installed.
set(CUVS_CLONE_ON_PIN ON)

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN ENABLE_NVTX BUILD_CUVS_C_LIBRARY)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if( NOT cuvs_ROOT AND NOT CPM_cuvs_SOURCE AND PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "amd-integration")
        message(STATUS "CUVS pinned tag found: ${PKG_PINNED_TAG}. Cloning locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    endif()

    set(CUVS_COMPONENTS "")
    if(PKG_BUILD_CUVS_C_LIBRARY)
        string(APPEND CUVS_COMPONENTS " c_api")
    endif()
    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-examples-exports
            INSTALL_EXPORT_SET  cuvs-examples-exports
            COMPONENTS ${CUVS_COMPONENTS}
            CPM_ARGS
            GIT_REPOSITORY https://$ENV{GITHUB_PASS}@github.com/${PKG_FORK}/hipVS.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
            "BUILD_C_LIBRARY ${PKG_BUILD_CUVS_C_LIBRARY}"
            "BUILD_TESTS OFF"
            "CUVS_NVTX ${PKG_ENABLE_NVTX}"
            )
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different CUVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        CLONE_ON_PIN             ${CUVS_CLONE_ON_PIN}
        ENABLE_NVTX              OFF
        BUILD_CUVS_C_LIBRARY     ${BUILD_CUVS_C_LIBRARY}
)
