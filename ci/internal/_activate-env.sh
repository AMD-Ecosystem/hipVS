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

# Shared helpers: environment validation and build-environment activation.
# Meant to be sourced, not executed directly.
#
# Provides:
#   require_envs()                    - fails with a clear error if any named
#                                       variable is unset or empty
#   activate_hipvs_venv()             - activates the dev venv build-dev.sh creates
#   activate_hipvs_conda_env()        - activates HIPVS_CONDA_ENV when set
#   setup_hipblaslt_tensile_libpath() - TheRock hipBLASLt layout workaround

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


# Works around TheRock-based ROCm packaging hipBLASLt's Tensile kernel data
# under a per-architecture subdirectory (.../hipblaslt/library/<gfx_arch>/)
# instead of directly under .../hipblaslt/library/, where the hipBLASLt runtime
# expects the lazy-load index (TensileLibrary_lazy_<arch>.dat) and its kernels.
# Without this, any hipBLASLt matmul (e.g. ivf_pq's rotate_padded_centers)
# fails to load its kernels and aborts the process.
#
# When the per-arch layout is detected, point HIPBLASLT_TENSILE_LIBPATH at the
# matching subdir. Architecture-agnostic: globs for the lazy index rather than
# hardcoding a gfx target.
#
# No-op when:
#   - HIPBLASLT_TENSILE_LIBPATH is already set (respect an explicit override)
#   - the lazy index is already at the top level (standard ROCm layout)
#   - no per-arch subdir with a lazy index is found
setup_hipblaslt_tensile_libpath() {
    if [ -n "${HIPBLASLT_TENSILE_LIBPATH:-}" ]; then
        return 0
    fi

    local lib_root="${ROCM_PATH:-/opt/rocm}/lib/hipblaslt/library"
    [ -d "${lib_root}" ] || return 0

    # Standard ROCm layout: the lazy index sits at the top level. Nothing to do.
    # Match both *.dat (standard ROCm) and *.dat.zlib (TheRock compressed variant).
    if compgen -G "${lib_root}/TensileLibrary_lazy_*.dat" > /dev/null ||
       compgen -G "${lib_root}/TensileLibrary_lazy_*.dat.zlib" > /dev/null; then
        return 0
    fi

    # TheRock layout: the lazy index lives in a per-arch subdir. Prefer the
    # subdir matching this machine's GPU arch, falling back to the first subdir
    # that carries a lazy index.
    local arch=""
    if command -v rocminfo > /dev/null 2>&1; then
        arch="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]\+' || true)"
    fi
    if [ -z "${arch}" ]; then
        # AMDGPU_TARGETS may be a ;/,/space-separated list; take the first entry.
        arch="$(echo "${AMDGPU_TARGETS:-}" | tr ',;' '  ' | awk '{print $1}')"
    fi

    _has_lazy_index() {
        compgen -G "${1}TensileLibrary_lazy_*.dat" > /dev/null ||
        compgen -G "${1}TensileLibrary_lazy_*.dat.zlib" > /dev/null
    }

    if [ -n "${arch}" ] && _has_lazy_index "${lib_root}/${arch}/"; then
        export HIPBLASLT_TENSILE_LIBPATH="${lib_root}/${arch}"
    else
        local dir
        for dir in "${lib_root}"/*/; do
            if _has_lazy_index "${dir}"; then
                export HIPBLASLT_TENSILE_LIBPATH="${dir%/}"
                break
            fi
        done
    fi

    unset -f _has_lazy_index

    if [ -n "${HIPBLASLT_TENSILE_LIBPATH:-}" ]; then
        echo "Set HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH} (TheRock hipBLASLt layout WAR)"
    fi
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
