# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# cmake-format: off
# MIT License
#
# Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# cmake-format: on

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################

include(rapids-test)

if(PROJECT_IS_TOP_LEVEL)
  rapids_test_init()
endif()

rapids_cmake_install_lib_dir(lib_dir)
include(GNUInstallDirs)

include(${rapids-cmake-dir}/cpm/gtest.cmake)
rapids_cpm_gtest(BUILD_STATIC)

function(ConfigureTest)

  set(options)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH)

  cmake_parse_arguments(_CUVS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED _CUVS_TEST_GPUS AND NOT DEFINED _CUVS_TEST_PERCENT)
    set(_CUVS_TEST_GPUS 1)
    set(_CUVS_TEST_PERCENT 30)
  endif()
  if(NOT DEFINED _CUVS_TEST_GPUS)
    set(_CUVS_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _CUVS_TEST_PERCENT)
    set(_CUVS_TEST_PERCENT 100)
  endif()

  set(TEST_NAME ${_CUVS_TEST_NAME})

  add_executable(${TEST_NAME} ${_CUVS_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE cuvs::c_api
            GTest::gtest
            GTest::gtest_main
            $<IF:$<BOOL:${CUVSC_STATIC_CUVS_LIBRARY}>,$<COMPILE_ONLY:cuvs::cuvs_static>,cuvs::cuvs>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_C_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../${lib_dir}"
  )

  target_include_directories(
    ${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${CUVS_C_SOURCE_DIR}/tests>"
                         "$<BUILD_INTERFACE:${CUVS_C_SOURCE_DIR}/../cpp/tests>"
  )

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_CUVS_TEST_GPUS}
    PERCENT ${_CUVS_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

ConfigureTest(NAME INTEROP_TEST PATH core/interop.cu)
ConfigureTest(
  NAME DISTANCE_C_TEST PATH distance/run_pairwise_distance_c.c distance/pairwise_distance_c.cu
)
ConfigureTest(NAME BRUTEFORCE_C_TEST PATH neighbors/run_brute_force_c.c neighbors/brute_force_c.cu)
ConfigureTest(NAME IVF_FLAT_C_TEST PATH neighbors/run_ivf_flat_c.c neighbors/ann_ivf_flat_c.cu)
ConfigureTest(NAME IVF_PQ_C_TEST PATH neighbors/run_ivf_pq_c.c neighbors/ann_ivf_pq_c.cu)
ConfigureTest(NAME CAGRA_C_TEST PATH neighbors/ann_cagra_c.cu)
ConfigureTest(NAME MG_C_TEST PATH neighbors/run_mg_c.c neighbors/ann_mg_c.cu)
ConfigureTest(
  NAME ALL_NEIGHBORS_C_TEST PATH neighbors/run_all_neighbors_c.c neighbors/all_neighbors_c.cu
)

if(BUILD_CAGRA_HNSWLIB)
  ConfigureTest(NAME HNSW_C_TEST PATH neighbors/ann_hnsw_c.cu)
  target_link_libraries(HNSW_C_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(HNSW_C_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)
endif()

# ##################################################################################################
# * Validate C API ---------------------------------------------------------------

ConfigureTest(NAME cuvs_c_headers PATH core/headers.c)
ConfigureTest(NAME cuvs_c_test PATH core/c_api.c)
# <<<<<<< OLD CODE from a53604ef9 (10b0d1ff4) - COMMENTED OUT
# # <<<<<<< OLD CODE from 3894b6a36 (7e0daa4a3) - COMMENTED OUT
# # # <<<<<<< OLD CODE from d07a3b6b2 (04f402a00) - COMMENTED OUT
# # # target_link_libraries(cuvs_c_test PRIVATE hip::host)
# # # =======
# # target_link_libraries(cuvs_c_test PRIVATE CUDA::cudart_static)
# # # >>>>>>> END NEW CODE (04f402a00)
# # =======
# target_link_libraries(cuvs_c_test PRIVATE CUDA::cudart)
# # >>>>>>> END NEW CODE (7e0daa4a3)
# =======
target_link_libraries(cuvs_c_test PRIVATE CUDA::cudart_static)
# >>>>>>> END NEW CODE (10b0d1ff4)
ConfigureTest(NAME cuvs_c_neighbors_test PATH neighbors/c_api.c)

# ##################################################################################################
# * Validate installed headers are listed in `all.h`

include(cmake/header_check.cmake)
cuvs_c_add_header_check("${CUVS_C_SOURCE_DIR}" "cuvs/core/all.h" INSTALL_COMPONENT_SET testing)

# ##################################################################################################
# * Install Tests ---------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
  rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)
endif()
