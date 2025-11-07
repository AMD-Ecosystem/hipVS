#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

# MIT License
#
# Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(find_and_configure_nlohmann_json)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(nlohmann_json ${PKG_VERSION}
            GLOBAL_TARGETS      nlohmann_json::nlohmann_json
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/json.git
            GIT_TAG                ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(nlohmann_json_ADDED)
        message(VERBOSE "cuVS: Using nlohmann_json located in ${nlohmann_json_SOURCE_DIR}")
    else()
        message(VERBOSE "cuVS: Using nlohmann_json located in ${nlohmann_json_DIR}")
    endif()

endfunction()

find_and_configure_nlohmann_json(VERSION  3.12.0
        FORK             nlohmann
        PINNED_TAG       v3.12.0
        EXCLUDE_FROM_ALL ON
        )
