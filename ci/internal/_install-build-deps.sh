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

# Install build dependencies for hipVS.
# Meant to be sourced, not executed directly.
#
# The build runs with --no-build-isolation, so pip installs nothing from the
# packages' own requires tables: everything they declare has to be here.
# verify_build_requires() in _wheel-helpers.sh checks that per package.
#
# Required environment variables:
#   BUILD_ARTIFACTS_DIR -- directory containing pre-built dependency wheels
#   HIPVS_BUILD_TYPE    -- controls which dependencies to install
#
# Optional environment variables:
#   CMAKE_VERSION         -- pins cmake
#   ROCMDS_LOGGER_VERSION -- pins amd-rocmds-logger
#   HIPMM_VERSION         -- pins amd-libhipmm and amd-hipmm
#   HIPRAFT_VERSION       -- pins amd-libraft and amd-pylibraft
#                            CI sets these from the builder-tester
#                            _envs/rocmds/<branch> env file, and a bare checkout
#                            without them installs unpinned

set -euo pipefail

pip install --upgrade pip

case "${HIPVS_BUILD_TYPE:-}" in
  all|cpp_py|cpp_wheels|wheels|install_all|install_cpp_py|install_cpp_wheels)
    # python/cuvs's .pyx files cimport cuda.bindings.cyruntime from the
    # hip-python set, and its cmake resolves rmm and raft through the cmake
    # prefixes the hipMM and hipRAFT wheels carry.
    #
    # The rocm-bindings-* / hip-python-interop set stays unpinned: their version
    # leads with the ROCm version of the base image, which no env key names.
    pip install --no-deps \
        --find-links "${BUILD_ARTIFACTS_DIR}" \
        "amd-rocmds-logger${ROCMDS_LOGGER_VERSION:+==${ROCMDS_LOGGER_VERSION}}" \
        "amd-libhipmm${HIPMM_VERSION:+==${HIPMM_VERSION}}" \
        "amd-hipmm${HIPMM_VERSION:+==${HIPMM_VERSION}}" \
        "amd-libraft${HIPRAFT_VERSION:+==${HIPRAFT_VERSION}}" \
        "amd-pylibraft${HIPRAFT_VERSION:+==${HIPRAFT_VERSION}}" \
        rocm-bindings-core \
        rocm-bindings-hip \
        hip-python-interop
    ;;
esac

pip install wheel auditwheel Cython scikit-build-core patchelf
pip install --force-reinstall "cmake${CMAKE_VERSION:+==${CMAKE_VERSION}}"
pip install ninja
