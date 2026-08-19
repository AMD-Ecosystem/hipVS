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

# JIT-LTO bitcode targets, in the exact order of the ArchFamily enum in
# cpp/include/cuvs/detail/jit_lto/FragmentEntry.hpp. Targets listed in CUVS_JIT_LTO_ARCH_FAMILIES
# are actually built; the rest leave nullptr / 0 slots in the generated per-tag tables and
# hard-error at runtime if such a device is used.
#
# Most entries are LLVM gfxN-generic targets, which hold codegen at a family's common feature subset
# so one bitcode serves every member of that family. Two constraints govern this list:
#
# 1. A generic target is only valid for the member set LLVM documents for it (see
#    https://llvm.org/docs/AMDGPUUsage.html). gfx9-generic covers gfx900/902/904/906/909/90c only --
#    gfx908 and gfx90a are deliberately excluded, and gfx90a (MI200) therefore needs a *concrete*
#    bitcode target. gfx9-generic itself covers no hardware cuVS supports, so it is not built at all.
# 2. The runtime relink cannot repair a wrong choice here. hiprtcLink is passed --offload-arch=<live
#    device>, but that only re-stamps the code object's EF_AMDGPU_MACH; LLVM drives codegen from the
#    per-function "target-cpu"/"target-features" attributes frozen into the bitcode at -emit-llvm
#    time. Picking a target the device is not a member of yields an object whose target-ID feature
#    bits disagree with the device (e.g. gfx9-generic has no sramecc, so the object claims
#    sramecc-unsupported against a gfx90a:sramecc+ MI200) and ROCr refuses to load it -- surfacing as
#    hipErrorNotFound from hipModuleGetFunction.
set(_JIT_LTO_ALL_FAMILY_ARCHES "gfx90a;gfx9-4-generic;gfx11-generic;gfx12-generic")
set(_JIT_LTO_ALL_FAMILY_SUFFIXES "gfx90a;gfx9_4_generic;gfx11_generic;gfx12_generic")

# Map a concrete gfx architecture name (e.g. "gfx1151", possibly with a ":sramecc" suffix) to the
# JIT-LTO bitcode target that covers it. Mirrors arch_family_from_gcn_arch() in
# cpp/src/detail/jit_lto/AlgorithmPlanner.cpp -- keep the two in sync. Sets <out_var> to that target
# (e.g. "gfx11-generic") or to "" if no built target covers the arch.
#
# Membership is matched exactly, never by gfx prefix: gfx11-7-generic (gfx117x) and gfx12-5-generic
# (gfx125x) are separate families from gfx11-generic / gfx12-generic, so a prefix match would
# silently pick a generic the part is not a member of. Neither of those two families is built here
# -- this ROCm ships no device library for them.
function(cuvs_jit_lto_arch_to_family out_var arch)
  # Strip feature suffix ("gfx1151:sramecc+" -> "gfx1151").
  string(REGEX REPLACE ":.*$" "" _a "${arch}")
  set(_family "")
  if(_a STREQUAL "gfx90a")
    # MI200. No LLVM generic target covers gfx90a, so this slot is a concrete arch.
    set(_family "gfx90a")
  elseif(_a MATCHES "^gfx(942|950)$")
    # MI300 (gfx942) / MI350, MI355 (gfx950).
    set(_family "gfx9-4-generic")
  elseif(_a MATCHES "^gfx(110[0-3]|115[0-3])$")
    # RDNA3 / RDNA3.5.
    set(_family "gfx11-generic")
  elseif(_a MATCHES "^gfx120[01]$")
    # RDNA4.
    set(_family "gfx12-generic")
  endif()
  set(${out_var}
      "${_family}"
      PARENT_SCOPE
  )
endfunction()

function(add_jit_lto_kernel kernel_target)
  set(options)
  set(one_value KERNEL_FILE)
  set(multi_value LINK_LIBRARIES)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

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

  # Match the optimization/assert configuration of the main library so the pre-compiled bitcode
  # makes the same (arch-correct) preprocessor decisions, e.g. libhipcxx TSC clock-rate selection
  # guarded by NDEBUG.
  set(_jit_lto_config_flags -O3 -DNDEBUG)

  # Build one bitcode + embedded header per *enabled* family. Each enabled family produces its own
  # .bc and a header defining a uniquely named symbol; the generated register_fatbin.cpp then
  # #includes all enabled family headers and assembles the per-family lookup tables.
  foreach(_family_arch _family_suffix IN ZIP_LISTS _JIT_LTO_ALL_FAMILY_ARCHES
                       _JIT_LTO_ALL_FAMILY_SUFFIXES
  )
    if(NOT "${_family_arch}" IN_LIST CUVS_JIT_LTO_ARCH_FAMILIES)
      continue()
    endif()

    set(_bc_file "${_JIT_LTO_KERNEL_FILE}.${_family_suffix}.bc")
    set(_family_header "${_JIT_LTO_KERNEL_FILE}.${_family_suffix}.h")
    set(_family_symbol "embedded_fatbin_${_family_suffix}")

    # Step 1: compile .cu → LLVM bitcode for this bitcode target (family generic or concrete arch).
    add_custom_command(
      OUTPUT "${_bc_file}"
      COMMAND
        "${CMAKE_HIP_COMPILER}" -x hip -emit-llvm -fgpu-rdc -c --cuda-device-only
        "--offload-arch=${_family_arch}" ${_jit_lto_config_flags} -include hip/hip_runtime.h
        -std=c++20 -fPIC -D__HIP_PLATFORM_AMD__
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:_jit_lto_include_resolver,INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:_jit_lto_include_resolver,COMPILE_DEFINITIONS>,PREPEND,-D>"
        -o "${_bc_file}" "${_JIT_LTO_KERNEL_FILE}"
      DEPENDS "${_JIT_LTO_KERNEL_FILE}"
      COMMENT "Compiling JIT bitcode (${_family_arch}): ${kernel_target}"
      COMMAND_EXPAND_LISTS
    )

    # Step 2: embed .bc → C header with a family-unique symbol name.
    add_custom_command(
      OUTPUT "${_family_header}"
      COMMAND "${Python3_EXECUTABLE}" "${_embed_bitcode_script}" "${_bc_file}" "${_family_header}"
              "--symbol" "${_family_symbol}"
      DEPENDS "${_bc_file}" "${_embed_bitcode_script}"
      COMMENT "Embedding bitcode (${_family_arch}): ${kernel_target}"
      VERBATIM
    )
  endforeach()
endfunction()

function(process_jit_lto_matrix_entry source_list_var)
  set(options)
  set(one_value NAME_FORMAT KERNEL_INPUT_FILE OUTPUT_DIRECTORY FRAGMENT_TAG_FORMAT
                MATRIX_JSON_ENTRY
  )
  set(multi_value KERNEL_LINK_LIBRARIES FRAGMENT_TAG_HEADER_FILES)

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
  set(fatbin_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_fatbin.cpp")
  configure_file("${_JIT_LTO_KERNEL_INPUT_FILE}" "${kernel_file}" @ONLY)

  # Build the per-family template substitutions for register_fatbin.cpp.in. The data/length
  # initializers must list *every* ArchFamily slot in enum order: enabled families reference their
  # embedded symbol, disabled ones are nullptr / 0 (and hard-error at runtime if such a device is
  # used).
  set(fatbin_family_includes "")
  set(fatbin_data_initializer "")
  set(fatbin_length_initializer "")
  set(family_header_files "")
  foreach(_family_arch _family_suffix IN ZIP_LISTS _JIT_LTO_ALL_FAMILY_ARCHES
                       _JIT_LTO_ALL_FAMILY_SUFFIXES
  )
    if("${_family_arch}" IN_LIST CUVS_JIT_LTO_ARCH_FAMILIES)
      set(_family_header "${kernel_file}.${_family_suffix}.h")
      set(_family_symbol "embedded_fatbin_${_family_suffix}")
      string(APPEND fatbin_family_includes "#include \"${_family_header}\"\n")
      string(APPEND fatbin_data_initializer "  ${_family_symbol},\n")
      string(APPEND fatbin_length_initializer "  sizeof(${_family_symbol}),\n")
      list(APPEND family_header_files "${_family_header}")
    else()
      string(APPEND fatbin_data_initializer "  nullptr,\n")
      string(APPEND fatbin_length_initializer "  0,\n")
    endif()
  endforeach()

  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/register_fatbin.cpp.in" "${fatbin_file}" @ONLY)

  add_jit_lto_kernel(
    ${kernel_target}
    KERNEL_FILE "${kernel_file}"
    LINK_LIBRARIES ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
  )
  list(APPEND ${source_list_var} ${family_header_files} "${fatbin_file}")
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
  set(multi_value KERNEL_LINK_LIBRARIES FRAGMENT_TAG_HEADER_FILES)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

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
    )
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
