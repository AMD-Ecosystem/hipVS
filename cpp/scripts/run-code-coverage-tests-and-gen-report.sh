#! /usr/bin/env bash

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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


#-------------------------------------------------------------------------------------------------
# This script runs the C++ and C API tests to collect code coverage data and generates an HTML report using
# `llvm-profdata` and `llvm-cov`. To enable code coverage instrumentation, build with the CMake
# option `ENABLE_CODE_COVERAGE=ON`. Run this script from inside the build directory, or set the
# environment variable `HIPVS_BUILD_DIR` to point to the build directory. The generated profile
# data will be located under `${HIPVS_BUILD_DIR}/profiles`, and the HTML report will be saved to
# `${HIPVS_BUILD_DIR}/coverage_report`.
#-------------------------------------------------------------------------------------------------


set -e

HIPVS_BUILD_DIR=${HIPVS_BUILD_DIR:="${PWD}"}
HIPVS_PROFILES_DIR=${HIPVS_BUILD_DIR}/profiles
HIPVS_COVERAGE_DATA=${HIPVS_BUILD_DIR}/hipvs-coverage.profdata
HIPVS_COVERAGE_REPORT_PATH=${HIPVS_BUILD_DIR}/coverage_report

mkdir -p "$HIPVS_PROFILES_DIR"

# Run the C++ and C API tests
BINARIES=("${HIPVS_BUILD_DIR}/libcuvs.so" "${HIPVS_BUILD_DIR}/c/libcuvs_c.so")
for GTEST_DIR in "${HIPVS_BUILD_DIR}/gtests" "${HIPVS_BUILD_DIR}/c/gtests"; do
    if [ -d "${GTEST_DIR}" ]; then
        cd "${GTEST_DIR}"
        for test_exec in *; do
            if [[ -x "$test_exec" ]]; then
                echo "Running ./${test_exec} ..."
                BINARIES+=("${GTEST_DIR}/${test_exec}")
                export LLVM_PROFILE_FILE="${HIPVS_PROFILES_DIR}/${test_exec}.profraw"
                if ! ./"$test_exec"; then
                    echo "Test ${test_exec} failed"
                    have_failures=1
                fi
            else
                echo "Skipping non-executable file: ${test_exec}"
            fi
        done
    else
        echo "Test directory not found at ${GTEST_DIR}"
        # shellcheck disable=SC2034
        have_failures=1
    fi
done

set -x

# Use the LLVM tools matching ROCm's clang, which wrote the profiles.
export PATH="${ROCM_PATH:-/opt/rocm}/lib/llvm/bin:${PATH}"

llvm-profdata merge -sparse "${HIPVS_PROFILES_DIR}"/*.profraw -o "${HIPVS_COVERAGE_DATA}"

mkdir -p "$HIPVS_COVERAGE_REPORT_PATH"

OBJECTS="$(printf ' --object=%s' "${BINARIES[@]}")"
LLVM_COV_OPTIONS="--format=html --show-line-counts-or-regions --Xdemangler c++filt"
# shellcheck disable=SC2086
llvm-cov show \
    ${OBJECTS} ${LLVM_COV_OPTIONS} \
    --instr-profile="${HIPVS_COVERAGE_DATA}" \
    --output-dir="${HIPVS_COVERAGE_REPORT_PATH}" \
    --ignore-filename-regex="${HIPVS_BUILD_DIR}"/_deps
