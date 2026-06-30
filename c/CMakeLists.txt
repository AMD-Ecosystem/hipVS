# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# MIT License
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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

# NOTE(HIP/AMD): make *.cu files HIP lang files
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/../overrides.cmake")

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  CUVS_C
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX C
)
set(CMAKE_INSTALL_MESSAGE LAZY)

# Disable C++20 module scanning (not using modules, just C++20 features)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuvs_c shared libraries" ON)
option(BUILD_TESTS "Build cuvs unit-tests" ON)

option(BUILD_CAGRA_HNSWLIB "Build CAGRA+hnswlib interface" ON)
option(BUILD_MG_ALGOS "Build with multi-GPU support" ON)

option(CUVSC_STATIC_CUVS_LIBRARY "Link against statical version of libcuvs" OFF)

# Check if cuVS is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuvs package.
if(NOT TARGET cuvs::cuvs)
  if(CUVSC_STATIC_CUVS_LIBRARY)
    find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS cuvs_static)
  else()
    find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS cuvs_shared)
  endif()
else()
  set(BUILDING_FROM_CUVS ON)
endif()

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CUVS_CXX_FLAGS)
  include(../cpp/cmake/modules/ConfigureHIP.cmake)
endif()

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

# add third party dependencies using CPM
include(../cpp/cmake/thirdparty/get_dlpack.cmake)

if(BUILD_CAGRA_HNSWLIB)
  include(../cpp/cmake/thirdparty/get_hnswlib.cmake)
endif()

if(BUILD_MG_ALGOS AND CUVSC_STATIC_CUVS_LIBRARY)
  rapids_find_generate_module(
    NCCL
    HEADER_NAMES nccl.h
    LIBRARY_NAMES nccl
  )
  find_package(NCCL REQUIRED)
endif()

# ##################################################################################################
# * cuvs_c -------------------------------------------------------------------------------
add_library(
  cuvs_c SHARED
  src/core/c_api.cpp
  src/cluster/kmeans.cpp
  src/neighbors/brute_force.cpp
  src/neighbors/ivf_flat.cpp
  src/neighbors/ivf_pq.cpp
  src/neighbors/cagra.cpp
  $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:src/neighbors/hnsw.cpp>
  $<$<BOOL:${BUILD_MG_ALGOS}>:src/neighbors/mg_ivf_pq.cpp>
  $<$<BOOL:${BUILD_MG_ALGOS}>:src/neighbors/mg_ivf_flat.cpp>
  $<$<BOOL:${BUILD_MG_ALGOS}>:src/neighbors/mg_cagra.cpp>
  src/neighbors/nn_descent.cpp
  src/neighbors/vamana.cpp
  src/neighbors/refine.cpp
  src/neighbors/tiered_index.cpp
  src/neighbors/all_neighbors.cpp
  src/preprocessing/quantize/binary.cpp
  src/preprocessing/quantize/pq.cpp
  src/preprocessing/quantize/scalar.cpp
  src/distance/pairwise_distance.cpp
)
add_library(cuvs::c_api ALIAS cuvs_c)
set_target_properties(
  cuvs_c
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             EXPORT_NAME c_api
)

# Setup the VERSION and SOVERSION of our library
#
# Long term goal is that every release that breaks the C ABI will update `cuvs_abi_version` allowing
# us to have multiple releases of libcuvs_c that don't need consumers to rebuild against
if(PROJECT_IS_TOP_LEVEL)
  include(../cpp/cmake/modules/determine_cuvs_abi_version.cmake)
  determine_cuvs_abi_version(${RAPIDS_VERSION} MAJOR abi_major MINOR abi_minor)
  set_target_properties(cuvs_c PROPERTIES VERSION "${abi_major}.${abi_minor}")
  set_target_properties(cuvs_c PROPERTIES SOVERSION "${abi_major}")
endif()

# Encode what optional components we are building
set(CUVS_BUILD_CAGRA_HNSWLIB ${BUILD_CAGRA_HNSWLIB})
set(CUVS_BUILD_MG_ALGOS ${BUILD_MG_ALGOS})
# cmake-lint: disable=E1126
file(
  CONFIGURE
  OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/include/cuvs/core/c_config.h"
  CONTENT
  [=[
  #pragma once
  #ifndef CUVS_BUILD_CAGRA_HNSWLIB
    #cmakedefine CUVS_BUILD_CAGRA_HNSWLIB
  #endif
  #ifndef CUVS_BUILD_MG_ALGOS
    #cmakedefine CUVS_BUILD_MG_ALGOS
  #endif
]=]
)
target_compile_definitions(
  cuvs_c
  PUBLIC $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:CUVS_BUILD_CAGRA_HNSWLIB>
         $<$<BOOL:${BUILD_MG_ALGOS}>:CUVS_BUILD_MG_ALGOS>
  PRIVATE __HIP_PLATFORM_AMD__=1
)

target_compile_options(cuvs_c PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>")
# Force-include libhipcxx's HIP compatibility shim so cuda* identifiers (used by libhipcxx headers)
# are mapped to hip* before any template instantiation sees them. Without this, CXX-compiled code
# hits missing hipDeviceAttribute* enum values.
find_file(
  _LIBHIPCXX_AMD_RUNTIME "libhipcxx/__amd/cuda_runtime.h"
  PATHS ${libhipcxx_SOURCE_DIR}/include
  NO_DEFAULT_PATH
)
if(_LIBHIPCXX_AMD_RUNTIME)
  target_compile_options(cuvs_c PRIVATE "-include${_LIBHIPCXX_AMD_RUNTIME}")
endif()

target_include_directories(
  cuvs_c
  PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(
  cuvs_c
  PRIVATE # we don't want any dependencies exported
          $<IF:$<BOOL:${CUVSC_STATIC_CUVS_LIBRARY}>,cuvs::cuvs_static,cuvs::cuvs> #
          $<COMPILE_ONLY:raft::raft> # enforce we shouldn't use raft symbols
          $<TARGET_NAME_IF_EXISTS:hnswlib::hnswlib>
)

# ##################################################################################################
# * install targets-----------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL)
  rapids_cmake_install_lib_dir(lib_dir)
  include(GNUInstallDirs)
  include(CPack)

  # Add CUDAToolkit as an export dependency
  rapids_export_package(INSTALL CUDAToolkit cuvs-exports)
  rapids_export_package(BUILD CUDAToolkit cuvs-exports)

  install(
    TARGETS cuvs_c
    DESTINATION ${lib_dir}
    COMPONENT c_api
    EXPORT cuvs-c-exports
  )
  install(
    DIRECTORY include/cuvs
    COMPONENT c_api
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuvs/core/c_config.h
    COMPONENT c_api
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuvs/core/
  )

  include(../cpp/cmake/modules/generate_cuvs_export.cmake)
  generate_cuvs_export(CLIB COMPONENTS c_api EXPORT_SETS cuvs-c-exports)

endif()

# ##################################################################################################
# * build tests ----------------------------------------------------
if(BUILD_TESTS)
  include(rapids-hip)
  rapids_hip_init_architectures(CUVS_C)
  enable_language(HIP)
  find_package(hip REQUIRED)

  enable_testing()
  add_subdirectory(tests)
endif()
