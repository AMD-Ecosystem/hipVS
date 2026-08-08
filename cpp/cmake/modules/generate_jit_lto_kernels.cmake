# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
# cmake-format: off
# MIT License
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

set(_embed_bitcode_script "${CMAKE_CURRENT_LIST_DIR}/embed_bitcode.py")

function(add_jit_lto_kernel kernel_target)
  set(options)
  set(one_value KERNEL_FILE FATBIN_HEADER_FILE)
  set(multi_value LINK_LIBRARIES EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  set(bc_file "${_JIT_LTO_KERNEL_FILE}.bc")

  # Filter out nvcc-only options that have no AMDGPU equivalent
  set(_filtered_opts)
  foreach(_opt IN LISTS _JIT_LTO_EXTRA_COMPILE_OPTIONS)
    if(NOT _opt MATCHES "^--maxrregcount" AND NOT _opt MATCHES "^-Xfatbin")
      list(APPEND _filtered_opts "${_opt}")
    endif()
  endforeach()

  # Create a helper target to resolve transitive include directories from all linked dependencies
  # (rmm, raft, CCCL, etc.).  We never build this target; we only read its INCLUDE_DIRECTORIES
  # property at generation time.
  if(NOT TARGET _jit_lto_include_resolver)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_jit_lto_stub.cpp" "")
    add_library(
      _jit_lto_include_resolver OBJECT EXCLUDE_FROM_ALL
                                       "${CMAKE_CURRENT_BINARY_DIR}/_jit_lto_stub.cpp"
    )
    target_link_libraries(_jit_lto_include_resolver PRIVATE jit_lto_kernel_usage_requirements)
  endif()

  # Step 1: compile .cu → LLVM bitcode via hipcc
  add_custom_command(
    OUTPUT "${bc_file}"
    COMMAND
      "${HIPCC}" -x hip -emit-llvm -fgpu-rdc -c --cuda-device-only -include hip/hip_runtime.h
      -std=c++20 -fPIC -D__HIP_PLATFORM_AMD__ ${_filtered_opts}
      "$<LIST:TRANSFORM,$<TARGET_PROPERTY:_jit_lto_include_resolver,INCLUDE_DIRECTORIES>,PREPEND,-I>"
      "$<LIST:TRANSFORM,$<TARGET_PROPERTY:_jit_lto_include_resolver,COMPILE_DEFINITIONS>,PREPEND,-D>"
      -o "${bc_file}" "${_JIT_LTO_KERNEL_FILE}"
    DEPENDS "${_JIT_LTO_KERNEL_FILE}"
    COMMENT "Compiling JIT bitcode: ${kernel_target}"
    COMMAND_EXPAND_LISTS
  )

  # Step 2: embed .bc → C header
  add_custom_command(
    OUTPUT "${_JIT_LTO_FATBIN_HEADER_FILE}"
    COMMAND "${Python3_EXECUTABLE}" "${_embed_bitcode_script}" "${bc_file}"
            "${_JIT_LTO_FATBIN_HEADER_FILE}"
    DEPENDS "${bc_file}" "${_embed_bitcode_script}"
    COMMENT "Embedding bitcode: ${kernel_target}"
    VERBATIM
  )
endfunction()

function(process_jit_lto_matrix_entry source_list_var)
  set(options)
  set(one_value NAME_FORMAT KERNEL_INPUT_FILE OUTPUT_DIRECTORY FRAGMENT_TAG_FORMAT
                MATRIX_JSON_ENTRY
  )
  set(multi_value KERNEL_LINK_LIBRARIES FRAGMENT_TAG_HEADER_FILES KERNEL_EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_JIT_LTO_MATRIX_JSON_ENTRY}")
  string(CONFIGURE "${_JIT_LTO_NAME_FORMAT}" kernel_name @ONLY)
  string(CONFIGURE "${_JIT_LTO_FRAGMENT_TAG_FORMAT}" fragment_tag @ONLY)

  set(fragment_tag_header_files "")
  foreach(header_file IN LISTS _JIT_LTO_FRAGMENT_TAG_HEADER_FILES)
    if(NOT header_file MATCHES "^(\".*\"|<.*>)$")
      set(header_file "\"${header_file}\"")
    endif()
    string(APPEND fragment_tag_header_files "#include ${header_file}\n")
  endforeach()

  set(kernel_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_kernel.cu")
  set(kernel_target "${kernel_name}_kernel")
  set(fatbin_header_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_fatbin.h")
  set(fatbin_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_fatbin.cpp")
  configure_file("${_JIT_LTO_KERNEL_INPUT_FILE}" "${kernel_file}" @ONLY)
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/register_fatbin.cpp.in" "${fatbin_file}" @ONLY)

  add_jit_lto_kernel(
    ${kernel_target}
    KERNEL_FILE "${kernel_file}"
    FATBIN_HEADER_FILE "${fatbin_header_file}"
    LINK_LIBRARIES ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
    EXTRA_COMPILE_OPTIONS ${_JIT_LTO_KERNEL_EXTRA_COMPILE_OPTIONS}
  )
  list(APPEND ${source_list_var} "${fatbin_header_file}" "${fatbin_file}")
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_jit_lto_kernels source_list_var)
  set(options)
  set(one_value NAME_FORMAT MATRIX_JSON_FILE MATRIX_JSON_STRING KERNEL_INPUT_FILE
                FRAGMENT_TAG_FORMAT OUTPUT_DIRECTORY
  )
  set(multi_value KERNEL_LINK_LIBRARIES FRAGMENT_TAG_HEADER_FILES KERNEL_EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  find_program(
    HIPCC
    NAMES hipcc
    PATHS /opt/rocm/bin ENV ROCM_PATH
    PATH_SUFFIXES bin REQUIRED
  )

  if(_JIT_LTO_MATRIX_JSON_FILE)
    set_property(
      DIRECTORY
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${_JIT_LTO_MATRIX_JSON_FILE}"
      APPEND
    )
    compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_JIT_LTO_MATRIX_JSON_FILE}")
  else()
    compute_matrix_product(matrix_product MATRIX_JSON_STRING "${_JIT_LTO_MATRIX_JSON_STRING}")
  endif()

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_jit_lto_matrix_entry(
      "${source_list_var}"
      NAME_FORMAT "${_JIT_LTO_NAME_FORMAT}"
      KERNEL_INPUT_FILE "${_JIT_LTO_KERNEL_INPUT_FILE}"
      FRAGMENT_TAG_FORMAT "${_JIT_LTO_FRAGMENT_TAG_FORMAT}"
      FRAGMENT_TAG_HEADER_FILES ${_JIT_LTO_FRAGMENT_TAG_HEADER_FILES}
      OUTPUT_DIRECTORY "${_JIT_LTO_OUTPUT_DIRECTORY}"
      MATRIX_JSON_ENTRY "${matrix_json_entry}"
      KERNEL_LINK_LIBRARIES ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
      KERNEL_EXTRA_COMPILE_OPTIONS ${_JIT_LTO_KERNEL_EXTRA_COMPILE_OPTIONS}
    )
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
