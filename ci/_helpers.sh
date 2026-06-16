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

set -euo pipefail

require_envs() {
    local missing=0
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            echo "ERROR: required environment variable ${var} is not set." >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo "ERROR: missing required environment variables (see above). Aborting." >&2
        exit 1
    fi
}


activate_hipvs_venv() {
    . "${INSTALL_DIR}/venv/hipvs_dev/bin/activate"
}


# Activates the conda env named by HIPVS_CONDA_ENV, if set.
#
# Returns 0 (and activates the env) when HIPVS_CONDA_ENV is set, 1 otherwise so
# callers can fall back to activate_hipvs_venv.
#
# Required environment variables (when HIPVS_CONDA_ENV is set):
#   HIPVS_CONDA_ENV -- name of the conda environment to activate
#   CONDA_DIR       -- path to the conda installation
activate_hipvs_conda_env() {
    if [ -n "${HIPVS_CONDA_ENV:-}" ]; then
        set +u
        source "${CONDA_DIR}/etc/profile.d/conda.sh"
        conda activate "${HIPVS_CONDA_ENV}"
        set -u

        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
        return 0
    fi
    return 1
}
