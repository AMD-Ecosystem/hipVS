# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

# Write the bin2c-replacement CMake script once at configure time.  Invoked at
# build time via  cmake -P <script> -DINPUT_FILE=... -DOUTPUT_FILE=...
# Produces:  static const uint8_t embedded_fatbin[] = { 0xNN, ... };
set(_bin2c_script "${CMAKE_CURRENT_BINARY_DIR}/_embed_bitcode.cmake")
file(
  WRITE "${_bin2c_script}"
  [=[
file(READ "${INPUT_FILE}" _hex HEX)
string(LENGTH "${_hex}" _hex_len)
set(_c_hex "")
math(EXPR _last "${_hex_len} - 2")
if(_last GREATER_EQUAL 0)
  foreach(_i RANGE 0 ${_last} 2)
    string(SUBSTRING "${_hex}" ${_i} 2 _byte)
    string(APPEND _c_hex "0x${_byte},")
  endforeach()
endif()
file(WRITE "${OUTPUT_FILE}"
  "static const uint8_t embedded_fatbin[] = {\n${_c_hex}\n};\n"
)
]=]
)

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

  # Create a helper target to resolve transitive include directories from all
  # linked dependencies (rmm, raft, CCCL, etc.).  We never build this target;
  # we only read its INCLUDE_DIRECTORIES property at generation time.
  if(NOT TARGET _jit_lto_include_resolver)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_jit_lto_stub.cpp" "")
    add_library(_jit_lto_include_resolver OBJECT EXCLUDE_FROM_ALL
                "${CMAKE_CURRENT_BINARY_DIR}/_jit_lto_stub.cpp")
    target_link_libraries(_jit_lto_include_resolver PRIVATE jit_lto_kernel_usage_requirements)
  endif()

  # Step 1: compile .cu → LLVM bitcode via hipcc
  add_custom_command(
    OUTPUT "${bc_file}"
    COMMAND
      "${HIPCC}"
      -x hip
      -emit-llvm
      -fgpu-rdc
      -c
      --cuda-device-only
      -include hip/hip_runtime.h
      -std=c++20
      -fPIC
      -D__HIP_PLATFORM_AMD__
      ${_filtered_opts}
      "$<LIST:TRANSFORM,$<TARGET_PROPERTY:_jit_lto_include_resolver,INCLUDE_DIRECTORIES>,PREPEND,-I>"
      "$<LIST:TRANSFORM,$<TARGET_PROPERTY:_jit_lto_include_resolver,COMPILE_DEFINITIONS>,PREPEND,-D>"
      -o "${bc_file}"
      "${_JIT_LTO_KERNEL_FILE}"
    DEPENDS "${_JIT_LTO_KERNEL_FILE}"
    COMMENT "Compiling JIT bitcode: ${kernel_target}"
    COMMAND_EXPAND_LISTS
  )

  # Step 2: embed .bc → C header via CMake script
  add_custom_command(
    OUTPUT "${_JIT_LTO_FATBIN_HEADER_FILE}"
    COMMAND
      "${CMAKE_COMMAND}"
      "-DINPUT_FILE=${bc_file}"
      "-DOUTPUT_FILE=${_JIT_LTO_FATBIN_HEADER_FILE}"
      -P "${_bin2c_script}"
    DEPENDS "${bc_file}"
    COMMENT "Embedding bitcode: ${kernel_target}"
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

  find_program(
    HIPCC
    NAMES hipcc
    PATHS /opt/rocm/bin ENV ROCM_PATH
    PATH_SUFFIXES bin
    REQUIRED
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
