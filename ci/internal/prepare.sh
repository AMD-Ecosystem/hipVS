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

# Prepare the build environment: install OS packages and cmake.
#
# Required environment variables:
#   INSTALL_DIR, CMAKE_VERSION

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_activate-env.sh"
require_envs INSTALL_DIR CMAKE_VERSION

os_id=$(. /etc/os-release; echo "$ID")
os_id_like=$(. /etc/os-release; echo "$ID_LIKE")

if [[ "${os_id}" == *"ubuntu"* ]]; then
    apt install -y libopenblas-dev ninja-build libsuitesparse-dev patchelf libclang-dev curl
elif [[ "${os_id_like}" == *"rhel"* ]]; then
    dnf install -y openblas-devel lapack-devel ninja-build suitesparse-devel patchelf clang-devel curl
fi

if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "${HOME}/.cargo/env"
fi

if ! activate_hipvs_conda_env; then
    python3 -m venv --clear "${INSTALL_DIR}/venv/hipvs_dev"
    activate_hipvs_venv
fi
pip install --upgrade pip
pip install --force-reinstall cmake${CMAKE_VERSION:+==${CMAKE_VERSION}}
pip install setuptools scikit-build-core auditwheel cython pytest "numba<0.62" fastrlock
