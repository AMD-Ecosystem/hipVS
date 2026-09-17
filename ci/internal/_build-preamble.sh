# shellcheck shell=bash
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

# Build preamble: sets up the build environment for hipVS.
# Meant to be sourced, not executed directly.
#
# Makes available locally:
#   cmake_args     -- CPM source and compiler-launcher args shared by the C++
#                     build, the wheel builds and the examples build
#   install_prefix -- the resolved install prefix
#
# Exports:
#   PARALLEL_LEVEL, CMAKE_PREFIX_PATH, CCC_OVERRIDE_OPTIONS / CC when a
#   gcc-toolset is the image default, CMAKE_GENERATOR on AlmaLinux
#
# Required environment variables:
#   SRC_DIR, INSTALL_DIR, ROCM_PATH
#   CMAKE_VERSION -- the version of the cmake package to install, unless a
#                    conda environment supplies cmake
#
# Optional environment variables:
#   HIPVS_CONDA_ENV, CONDA_DIR, HIPVS_USE_VERSIONS_JSON,
#   SCCACHE_ENABLE, AMDGPU_TARGETS, MAX_JOBS
#   CMAKE_INSTALL_PREFIX -- overrides INSTALL_DIR as the install prefix;
#                           '<conda-prefix>' selects the active conda env
#   HIPVS_ENABLE_CK_BACKEND -- build the Composable Kernel distance backend
#                              (default: true); "false" passes
#                              -DBUILD_CK_BACKEND=OFF

set -euo pipefail

_PREAMBLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_PREAMBLE_DIR}/_activate-env.sh"

# Prefer the conda env (HIPVS_CONDA_ENV) when set; otherwise use the venv.
# cmake, cpack and ctest come from this venv, which the tests tarball ships.
if ! activate_hipvs_conda_env; then
    python3 --version
    python3 -m venv --clear "${INSTALL_DIR}/venv/hipvs_dev"
    activate_hipvs_venv
    pip install --upgrade pip
    pip install "cmake${CMAKE_VERSION:+==${CMAKE_VERSION}}"
fi
cmake --version && cpack --version && ctest --version

[[ "${HIPVS_USE_VERSIONS_JSON:-false}" == "false" ]] && no_use_versions_json=1

[[ "${SCCACHE_ENABLE:-false}" == "true" ]] && use_sccache=1

[[ "${HIPVS_ENABLE_CK_BACKEND:-true}" == "false" ]] && disable_ck_backend=1

# The job's install root is the prefix unless the caller overrides it, so a
# build that only gets the mounts still installs where the job expects.
if [[ "${CMAKE_INSTALL_PREFIX:-}" == "<conda-prefix>" ]]; then
    install_prefix="${CONDA_PREFIX}"
else
    install_prefix="${CMAKE_INSTALL_PREFIX:-${INSTALL_DIR}}"
fi

# The dependency checkouts CPM resolves from instead of downloading. The paths
# are the manifest's, and the dev pipelines use the same ones.
# shellcheck disable=SC2034  # consumed by the script that sources this preamble
cmake_args=(
    "-DCUVS_RAFT_CLONE_ON_PIN=OFF"
    ${no_use_versions_json:+"-DCPM_libhipcxx_SOURCE=${SRC_DIR}/libhipcxx"}
    ${no_use_versions_json:+"-DCPM_cuco_SOURCE=${SRC_DIR}/hipcollections"}
    ${no_use_versions_json:+"-DCPM_rmm_SOURCE=${SRC_DIR}/hipmm"}
    ${no_use_versions_json:+"-DCPM_raft_SOURCE=${SRC_DIR}/hipraft"}
    ${no_use_versions_json:+"-DCPM_rapids_logger_SOURCE=${SRC_DIR}/rocmds_logger"}
    ${use_sccache:+"-DCMAKE_HIP_COMPILER_LAUNCHER=sccache"}
    ${use_sccache:+"-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"}
    ${use_sccache:+"-DCMAKE_C_COMPILER_LAUNCHER=sccache"}
    ${disable_ck_backend:+"-DBUILD_CK_BACKEND=OFF"}
)

if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
    toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
    export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
    # note: in some shells, cmake is using /bin/{cc, c++} as compilers,
    # which may not be the compilers of the enabled toolchain.
    export CC=${toolchain}/bin/cc
fi

os_id=$(. /etc/os-release; echo "$ID")
if [[ "${os_id}" == *"almalinux"* ]]; then
    export CMAKE_GENERATOR="Unix Makefiles"
fi

# Build parallelism, in the two spellings it has to travel in.
#
# PARALLEL_LEVEL is the RAPIDS convention: build.sh reads it for its own
# `cmake --build -j`, so the C++ build has always honoured it.
#
# CMAKE_BUILD_PARALLEL_LEVEL is the one CMake itself honours, and it is the
# only one that reaches the wheel builds. scikit-build-core runs a bare
# `cmake --build <dir>` with no -j of its own and ignores PARALLEL_LEVEL (and
# classic scikit-build's SKBUILD_BUILD_OPTIONS, which build.sh still sets).
# Combined with the AlmaLinux branch above selecting Unix Makefiles -- whose
# default is one job, where Ninja would have defaulted to the host's core count
# and hidden this -- the wheels were compiling single-threaded no matter what
# MAX_JOBS said.
export PARALLEL_LEVEL=${MAX_JOBS:-$(nproc)}
export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL}
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}:${install_prefix}:${ROCM_PATH}/lib/cmake"
