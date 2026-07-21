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

# Run hipVS ANN benchmarks via the cuvs_bench Python framework.
#
# Required environment variables:
#   PROJECT_ID, SRC_DIR, BUILD_DIR, INSTALL_DIR, ARTIFACTS_DIR
#   CMAKE_INSTALL_PREFIX             -- install prefix used at build time (executables land in $CMAKE_INSTALL_PREFIX/bin/ann)
#   HIPVS_BENCHMARK_DATASET          -- dataset name (e.g. sift-128-euclidean, test-data)
#   HIPVS_BENCHMARK_ALGORITHMS       -- comma-separated algorithm list (e.g. cuvs_cagra)
#   HIPVS_BENCHMARK_TOP_K            -- number of nearest neighbors (e.g. 64)
#   HIPVS_BENCHMARK_BATCH_SIZE       -- search batch size (e.g. 512)
#   HIPVS_BENCHMARK_SEARCH_MODE      -- latency or throughput
#   HIPVS_BENCHMARK_CONFIG_GROUP     -- algorithm config group (e.g. base, test)

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_helpers.sh"
require_envs PROJECT_ID SRC_DIR BUILD_DIR INSTALL_DIR ARTIFACTS_DIR \
    CMAKE_INSTALL_PREFIX \
    HIPVS_BENCHMARK_DATASET HIPVS_BENCHMARK_ALGORITHMS \
    HIPVS_BENCHMARK_TOP_K HIPVS_BENCHMARK_BATCH_SIZE \
    HIPVS_BENCHMARK_SEARCH_MODE HIPVS_BENCHMARK_CONFIG_GROUP

if ! activate_hipvs_conda_env; then
    activate_hipvs_venv
fi

pip install click "matplotlib>=3.9" pandas pyyaml requests scikit-learn h5py

export BENCH_DIR="${BUILD_DIR}/${PROJECT_ID}/cpp/build/bench"
export RAPIDS_DATASET_ROOT_DIR="${BUILD_DIR}/datasets"
mkdir -p "${RAPIDS_DATASET_ROOT_DIR}"

DATASET_NAME="${HIPVS_BENCHMARK_DATASET}"

NORMALIZE_FLAG=""
if [[ "${DATASET_NAME}" == *-angular ]]; then
    NORMALIZE_FLAG="--normalize"
fi

python -m cuvs_bench.get_dataset --dataset "${DATASET_NAME}" ${NORMALIZE_FLAG}

if [[ -n "${NORMALIZE_FLAG}" ]]; then
    DATASET_NAME="${DATASET_NAME%-angular}-inner"
    echo "Using normalized dataset name: ${DATASET_NAME}"
fi

python -m cuvs_bench.run \
    --dataset "${DATASET_NAME}" \
    --algorithms "${HIPVS_BENCHMARK_ALGORITHMS}" \
    --groups "${HIPVS_BENCHMARK_CONFIG_GROUP}" \
    --dataset-path "${RAPIDS_DATASET_ROOT_DIR}" \
    --executable-dir "${CMAKE_INSTALL_PREFIX}/bin/ann" \
    -k "${HIPVS_BENCHMARK_TOP_K}" \
    --batch-size "${HIPVS_BENCHMARK_BATCH_SIZE}" \
    --search-mode "${HIPVS_BENCHMARK_SEARCH_MODE}" \
    --force

python -m cuvs_bench.run --data-export \
    --dataset "${DATASET_NAME}" \
    --algorithms "${HIPVS_BENCHMARK_ALGORITHMS}" \
    --groups "${HIPVS_BENCHMARK_CONFIG_GROUP}" \
    --dataset-path "${RAPIDS_DATASET_ROOT_DIR}" \
    --executable-dir "${CMAKE_INSTALL_PREFIX}/bin/ann" \
    -k "${HIPVS_BENCHMARK_TOP_K}" \
    --batch-size "${HIPVS_BENCHMARK_BATCH_SIZE}" \
    --search-mode "${HIPVS_BENCHMARK_SEARCH_MODE}"

GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
UUID=$(cat /proc/sys/kernel/random/uuid)
mkdir -p "${ARTIFACTS_DIR}/${GPU_ARCH}/${UUID}"
cp -r "${RAPIDS_DATASET_ROOT_DIR}/${DATASET_NAME}/result" "${ARTIFACTS_DIR}/${GPU_ARCH}/${UUID}/"

bash "${SRC_DIR}/${PROJECT_ID}/python/cuvs_bench/report_generator/collect_system_info.sh" "${ARTIFACTS_DIR}"
BENCHMARK_SYSTEM_INFO_SPEC_FILEPATH=$(find "${ARTIFACTS_DIR}" -name "system_info*.json")

BENCHMARK_RESULT_SPEC="${GPU_ARCH}:${RAPIDS_DATASET_ROOT_DIR}/${DATASET_NAME}/result"
BENCHMARK_SYSTEM_INFO_SPEC="${GPU_ARCH}:${BENCHMARK_SYSTEM_INFO_SPEC_FILEPATH}"

python "${SRC_DIR}/${PROJECT_ID}/python/cuvs_bench/report_generator/generate_html_report.py" \
    "${BENCHMARK_RESULT_SPEC}" \
    --system-info \
    "${BENCHMARK_SYSTEM_INFO_SPEC}" \
    -o "${ARTIFACTS_DIR}/report.html"
