#!/bin/bash

# Copyright (c) 2023-2024, NVIDIA CORPORATION.

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
#
# cuvs empty project template build script

#-----------------------------------------------------------------------------
# The following environment variables can be set to configure the build
# PARALLEL_LEVEL: Maximum number of parallel build jobs
# BUILD_TYPE: CMake build type (default: Release)
# CUVS_REPO_REL: The path to locally available hipVS sources. If unspecified, the
#                source is downloaded from github. Please refer to
#                'cmake/get_cuvs.cmake' for further configuration options.
# HIPVS_CMAKE_HIP_ARCHITECTURES: The GPU architectures to build for. Possible values are:
#                                1. A list of gpu archs, for example, "gfx90a;gfx420;..."
#                                2. NATIVE to build for the build machines native gpu arch (default)
#                                3. RAPIDS to build for all oif the supported gpu archs
# EXTRA_CMAKE_ARGS: Extra options to be passed to CMake. Specify in the form of
#                   "-D<CMAKE_VAR1>=v1 -D<CMAKE_VAR2>=v2 ..."
#

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg clean; then
  rm -rf c/build
  rm -rf cpp/build
  exit 0
fi

function gpuArch {

    if hasArg --allgpuarch && [[ -n $(echo $ARGS | { grep -E "\-\-gpu\-arch" || true; } ) ]]; then
        echo "Error: Cannot specify both --gpu-arch and --allgpuarch"
        echo "Use either:"
        echo "  --gpu-arch=\"gfx90a;gfx942\"    (for specific architectures)"
        echo "  --allgpuarch        (for all supported architectures)"
        exit 1
    fi

    if [[ $(echo $ARGS | { grep -Eo "\-\-gpu\-arch" || true; } | wc -l ) -gt 1 ]]; then
        echo "Error: Multiple --gpu-arch options were provided. Please combine architectures into a single option."
        echo "Instead of: --gpu-arch=gfx90a --gpu-arch=gfx942"
        echo "Use:       --gpu-arch=\"gfx90a;gfx942\""
        exit 1
    fi

    if [[ -n $(echo $ARGS | { grep -E "\-\-gpu\-arch" || true; } ) ]]; then
        GPU_ARCH_ARG=$(echo $ARGS | { grep -Eo "\-\-gpu\-arch=.+( |$)" || true; })
        if [[ -n ${GPU_ARCH_ARG} ]]; then
            # Extract just the architecture value
            echo ${GPU_ARCH_ARG} | sed -e 's/--gpu-arch=//' -e 's/ .*//'
            return
        fi
    fi

    # Handle --allgpuarch
    if hasArg --allgpuarch; then
        echo "ROCMDS"
        return
    fi

    # Default to NATIVE
    echo "NATIVE"
}

# Set up build configuration
PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=${BUILD_TYPE:="Release"}
BUILD_DIR=build/
CUVS_REPO_REL=""
EXTRA_CMAKE_ARGS=""


HIPVS_CMAKE_HIP_ARCHITECTURES=$(gpuArch)
case ${HIPVS_CMAKE_HIP_ARCHITECTURES} in
    "ROCMDS") echo "Building for *ALL* supported GPU architectures..." ;;
    "NATIVE") echo "Building for the architecture of the GPU in the system..." ;;
    *) echo "Building for specified GPU architectures: ${CUVS_CMAKE_CUDA_ARCHITECTURES}" ;;
esac

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")
EXAMPLE_LANGS="c cpp"

if [[ ${CUVS_REPO_REL} != "" ]]; then
  CUVS_REPO_PATH="`readlink -f \"${CUVS_REPO_REL}\"`"
  echo "Using existing cuVS source tree at ${CUVS_REPO_PATH}"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_cuvs_SOURCE=${CUVS_REPO_PATH}"
fi

################################################################################
export CC=hipcc
export CXX=hipcc

build_example() {
  example_dir=${1}
  example_dir="${EXAMPLES_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  CMAKE_OVERRIDES_FILE="${EXAMPLES_DIR}/cmake/overrides.cmake"

  # Configure
  cmake -S ${example_dir} -B ${build_dir} \
        -DCMAKE_USER_MAKE_RULES_OVERRIDE=${CMAKE_OVERRIDES_FILE} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCUDA_BACKEND=OFF \
        -DCMAKE_HIP_ARCHITECTURES=${HIPVS_CMAKE_HIP_ARCHITECTURES} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ${EXTRA_CMAKE_ARGS}

  # Build
  cmake --build ${build_dir} -j${PARALLEL_LEVEL}
}

for lang in ${EXAMPLE_LANGS}; do
  build_example ${lang}
done
