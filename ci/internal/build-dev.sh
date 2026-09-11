#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Build the hipVS C++ library, its tests and ANN benchmarks, and optionally the
# C and C++ examples. The Python wheels are built by build-wheels.sh and the
# Rust bindings by build-rust.sh.
#
# Required environment variables:
#   HIPVS_BUILD_TYPE  -- build scope (cpp, cpp_py, cpp_wheels, examples, rust,
#                        all, and their install_* forms)
#   PROJECT_ID, SRC_DIR, BUILD_DIR, INSTALL_DIR, ROCM_PATH, AMDGPU_TARGETS,
#   BUILD_ARTIFACTS_DIR, CMAKE_INSTALL_PREFIX
#
# Optional environment variables:
#   HIPVS_USE_VERSIONS_JSON  -- fetch deps via versions.json instead of /src/ (default: false)
#   HIPVS_BUILD_TESTS        -- build C++ and C tests (default: true)
#   HIPVS_BUILD_BENCH_ANN    -- build ANN benchmarks (default: true)
#   SCCACHE_ENABLE           -- use sccache as compiler launcher (default: false)
#   CI_INTERNAL_CREATE_TESTS_TARBALL -- '1' packs the tests tarball the tester
#                                       images consume (default: 0).

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_activate-env.sh"
require_envs HIPVS_BUILD_TYPE PROJECT_ID SRC_DIR BUILD_DIR INSTALL_DIR ROCM_PATH AMDGPU_TARGETS BUILD_ARTIFACTS_DIR CMAKE_INSTALL_PREFIX

build_dir="${BUILD_DIR}/${PROJECT_ID}"
src_dir="${SRC_DIR}/${PROJECT_ID}"
tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

case "${CI_INTERNAL_CREATE_TESTS_TARBALL:-0}" in
    1 | true | yes | on) create_tests_tarball="1" ;;
    *) create_tests_tarball="" ;;
esac

# 'examples' builds against an already-built tree, so it is the one type that
# must not wipe it.
[[ "${HIPVS_BUILD_TYPE}" != "examples" ]] && build_cpp=1
[[ "${HIPVS_BUILD_TYPE}" == "examples" ]] && build_examples=1
[[ "${HIPVS_BUILD_TYPE}" == "all" ]] && build_examples=1
[[ "${HIPVS_BUILD_TYPE}" == "install_all" ]] && build_examples=1

if [ -n "${build_cpp:-}" ]; then
    rm -rf "${build_dir}" # clean build files (in case of rebuild)
    mkdir -p "$(dirname "${build_dir}")"
    cp -R "${src_dir}" "${build_dir}"
fi

cd "${build_dir}"
    . "${SCRIPT_DIR}/_build-preamble.sh"

    cmake_args+=("-DCMAKE_INSTALL_PREFIX=${install_prefix}")

if [ -n "${build_cpp:-}" ]; then
    # Only skip install for cpp-only builds; wheel builds require libcuvs installed.
    [[ "${HIPVS_BUILD_TYPE}" == "cpp" ]] && no_install=1

    build_sh_args=(clean libcuvs ${no_install:+"-n"})

    [[ "${HIPVS_BUILD_TESTS:-true}" == "true" ]] && build_sh_args+=(tests)
    [[ "${HIPVS_BUILD_BENCH_ANN:-true}" == "true" ]] && build_sh_args+=(bench-ann)

    [[ -n "${AMDGPU_TARGETS}" ]] && build_sh_args+=("--gpu-arch=${AMDGPU_TARGETS}")

    CC=hipcc CXX=hipcc bash build.sh "${build_sh_args[@]}" \
        ${cmake_args:+"--cmake-args=\"${cmake_args[*]}\""}
fi

if [ -n "${build_examples:-}" ]; then
    overrides_cmake=$(readlink -f overrides.cmake)

    # C and C++ examples both link the installed cuvs, so they build against
    # the prefix rather than the build tree.
    for example_lang in c cpp; do
        CC=hipcc CXX=hipcc CPM_DOWNLOAD_ALL=OFF \
            cmake \
            -DCMAKE_HIP_ARCHITECTURES="${AMDGPU_TARGETS}" \
            -Dcuvs_ROOT="${install_prefix}" \
            -DCMAKE_USER_MAKE_RULES_OVERRIDE="${overrides_cmake}" \
            "${cmake_args[@]}" \
            -S "examples/${example_lang}" \
            -B "examples/${example_lang}/build"
        cmake --build "examples/${example_lang}/build" --parallel "$(nproc)"
    done
fi

#### test archive

# The tester images extract this at / and drive ci/internal/test-*.sh out of
# the build tree it carries, so the archive keeps absolute paths. The whole
# build directory travels, not just cpp/build: the tester runs ctest against
# the prebuilt gtest binaries, pytest against python/cuvs, and reads the
# ci/internal/ scripts from the same copy. The cmake/ctest venv comes along
# because the build tree records that interpreter's absolute cmake path in its
# ctest configuration, unless a conda env stood in for the venv, in which case
# the tester brings its own cmake. Packing here is also why this script runs
# before build-wheels.sh, which recreates the venv with --clear.
if [ -n "${create_tests_tarball:-}" ]; then
    dev_venv="${INSTALL_DIR}/venv/hipvs_dev"
    [ -d "${dev_venv}" ] || dev_venv=""

    tar --exclude=.git -czf "${BUILD_ARTIFACTS_DIR}/${tests_tarball_name}" \
        "${build_dir}" ${dev_venv}
    du -sh "${BUILD_ARTIFACTS_DIR}/${tests_tarball_name}"
fi
