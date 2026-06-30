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

# Run hipVS test suites based on environment variables.
#
# Required environment variables:
#   PROJECT_ID, BUILD_DIR, INSTALL_DIR
#
# Optional environment variables (each defaults to false):
#   HIPVS_CPP_TESTS      -- run C++ and C tests via ctest
#   HIPVS_PYTHON_TESTS   -- run Python tests via pytest
#   HIPVS_RUST_TESTS     -- run Rust tests via cargo test
#   HIPVS_RUN_BENCH_ANN  -- run ANN benchmarks
#   HIPVS_RUN_EXAMPLES   -- run C and C++ examples

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_helpers.sh"
require_envs PROJECT_ID BUILD_DIR INSTALL_DIR

printenv

if ! activate_hipvs_conda_env; then
    activate_hipvs_venv
fi

# TheRock-based ROCm installs hipBLASLt's Tensile kernels under a per-arch
# subdir; point the runtime at them so hipBLASLt matmuls don't abort. No-op on
# standard ROCm layouts and when HIPBLASLT_TENSILE_LIBPATH is already set.
setup_hipblaslt_tensile_libpath

exit_code=0

if [[ "${HIPVS_CPP_TESTS:-false}" == "true" ]]; then
    echo "=== Running C++ tests ==="
    (cd "${BUILD_DIR}/${PROJECT_ID}/cpp/build/tests" && ctest --output-on-failure .) || exit_code=$?

    # The C tests are registered in a sibling build dir (cpp/build/c/tests), so the
    # C++ ctest invocation above does not pick them up. Run them here when present.
    # They are only built when HIPVS_BUILD_TESTS=true (-> BUILD_C_TESTS=ON).
    c_test_dir="${BUILD_DIR}/${PROJECT_ID}/cpp/build/c/tests"
    if [[ -f "${c_test_dir}/CTestTestfile.cmake" ]]; then
        echo "=== Running C tests ==="
        (cd "${c_test_dir}" && ctest --output-on-failure .) || exit_code=$?
    else
        echo "=== Skipping C tests (not built; no ${c_test_dir}/CTestTestfile.cmake) ==="
    fi
fi

if [[ "${HIPVS_PYTHON_TESTS:-false}" == "true" ]]; then
    echo "=== Running Python tests ==="
    pip install scikit-learn pytest-cov
    (cd "${BUILD_DIR}/${PROJECT_ID}" && pytest -s -v python/cuvs/) || exit_code=$?
fi

if [[ "${HIPVS_RUN_BENCH_ANN:-false}" == "true" ]]; then
    echo "=== Running ANN benchmarks ==="
    bash "${SCRIPT_DIR}/bench.sh" || exit_code=$?
fi

if [[ "${HIPVS_RUST_TESTS:-false}" == "true" ]]; then
    echo "=== Running Rust tests ==="
    if [ -f "${HOME}/.cargo/env" ]; then
        source "${HOME}/.cargo/env"
    fi
    rust_lib_prefix="${CMAKE_INSTALL_PREFIX:-/usr/local}"
    if [[ "${rust_lib_prefix}" == "<conda-prefix>" ]]; then
        rust_lib_prefix="${CONDA_PREFIX}"
    fi
    (cd "${BUILD_DIR}/${PROJECT_ID}/rust" && \
        LD_LIBRARY_PATH="${rust_lib_prefix}/lib:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}" cargo test) || exit_code=$?
fi

if [[ "${HIPVS_RUN_EXAMPLES:-false}" == "true" ]]; then
    echo "=== Running C examples ==="
    (cd "${BUILD_DIR}/${PROJECT_ID}/examples/c/build" && find . -maxdepth 1 -not -name "*.so*" -type f -executable -exec {} \;) || exit_code=$?

    echo "=== Running C++ examples ==="
    (cd "${BUILD_DIR}/${PROJECT_ID}/examples/cpp/build" && \
        find . -maxdepth 1 -not -name "*.so*" -not -name "*VAMANA_EXAMPLE*" -type f -executable -exec {} \; && \
        ./VAMANA_EXAMPLE "${BUILD_DIR}/${PROJECT_ID}/examples/cpp/src/vamana_example_input.bin" output.bin float 64 128 0.06 1) || exit_code=$?
fi

exit "${exit_code}"
