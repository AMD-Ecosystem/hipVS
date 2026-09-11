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

# Run the hipVS Rust test suite via cargo test.
#
# Required environment variables:
#   PROJECT_ID, BUILD_DIR, INSTALL_DIR
#
# Optional environment variables:
#   CMAKE_INSTALL_PREFIX -- where libcuvs.so was installed; '<conda-prefix>'
#                           selects the active conda env
#   HIPVS_CONDA_ENV, CONDA_DIR -- for conda-based environments

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_activate-env.sh"
require_envs PROJECT_ID BUILD_DIR INSTALL_DIR

if ! activate_hipvs_conda_env; then
    activate_hipvs_venv
fi

setup_hipblaslt_tensile_libpath

if [ -f "${HOME}/.cargo/env" ]; then
    source "${HOME}/.cargo/env"
fi

rust_lib_prefix="${CMAKE_INSTALL_PREFIX:-/usr/local}"
if [[ "${rust_lib_prefix}" == "<conda-prefix>" ]]; then
    rust_lib_prefix="${CONDA_PREFIX}"
fi

echo "=== Running Rust tests ==="
cd "${BUILD_DIR}/${PROJECT_ID}/rust"
LD_LIBRARY_PATH="${rust_lib_prefix}/lib:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}" cargo test
