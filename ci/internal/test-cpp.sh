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

# Run the hipVS C++ and C test suites via ctest, and optionally the examples.
#
# Interpreter-independent, so the manylinux tester runs this once beside the
# per-interpreter wheel matrix.
#
# Required environment variables:
#   PROJECT_ID, BUILD_DIR, INSTALL_DIR
#
# Optional environment variables:
#   HIPVS_RUN_EXAMPLES -- also run the built C and C++ examples (default: false)
#   HIPVS_CONDA_ENV, CONDA_DIR -- for conda-based environments

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_activate-env.sh"
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

echo "=== Running C++ tests ==="
(cd "${BUILD_DIR}/${PROJECT_ID}/cpp/build/tests" && ctest -V .) || exit_code=$?

# The C tests are registered in a sibling build dir (cpp/build/c/tests), so the
# C++ ctest invocation above does not pick them up. Run them here when present.
# They are only built when HIPVS_BUILD_TESTS=true (-> BUILD_C_TESTS=ON).
c_test_dir="${BUILD_DIR}/${PROJECT_ID}/cpp/build/c/tests"
if [[ -f "${c_test_dir}/CTestTestfile.cmake" ]]; then
    echo "=== Running C tests ==="
    (cd "${c_test_dir}" && ctest -V .) || exit_code=$?
else
    echo "=== Skipping C tests (not built; no ${c_test_dir}/CTestTestfile.cmake) ==="
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
