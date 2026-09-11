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

# Build the hipVS Rust bindings against the C++ library build-dev.sh produced.
#
# Required environment variables:
#   PROJECT_ID, SRC_DIR, BUILD_DIR, INSTALL_DIR

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_activate-env.sh"
require_envs PROJECT_ID SRC_DIR BUILD_DIR INSTALL_DIR

# shellcheck disable=SC2153  # BUILD_DIR is a job mount, not a typo
build_dir="${BUILD_DIR}/${PROJECT_ID}"

if [[ ! -d "${build_dir}" ]]; then
    cp -R "${SRC_DIR}/${PROJECT_ID}" "${build_dir}"
fi

cd "${build_dir}/rust"

if [ -f "${HOME}/.cargo/env" ]; then
    source "${HOME}/.cargo/env"
fi

cargo build --examples --lib
