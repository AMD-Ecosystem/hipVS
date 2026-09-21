# shellcheck shell=bash
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

# Shared test helper functions for hipVS CI.
# Meant to be sourced, not executed directly.
#
# Provides:
#   run_tests()          -- runs pytest per test file with CSV output and timeout
#   merge_test_results() -- merges per-file CSV results into a single file
#   coverage_enabled / begin_llvm_profiling / export_llvm_cov_json / run_pytest_cov
#
# Prerequisites: pytest, pytest-csv, pytest-timeout must be installed.

set -euo pipefail

# Runs all tests individually given the input folder (1st arg),
# the output folder (2nd arg), and a timeout value (3rd arg)
# that should be chosen so that long running tests
# complete but hanging tests are aborted.
# The 4th argument can be used to append additional
# pytest arguments. Do not forget to use quotes if there are
# more than one arguments.
# The 5th argument can be used to append a suffix to the CSV filename.
# The 6th argument can be used to append a suffix to the test ID in CSV output.
function run_tests() {
    set +e # continue on error
    local pytest_args=${4:-""}
    local suffix=${5:-""}
    local id_suffix=${6:-""}
    find "$1" -name "test_*.py" \
      -printf "run_tests: found test file: %p\n" \
      -exec bash -c '
        csv_file="$2/$7${5}.csv"
        mkdir -p "$(dirname "$csv_file")"
        pytest $4 -v --timeout $3 "$7" \
        --continue-on-collection-errors \
        --csv "$csv_file" --csv-columns file,id,status,message
        if [[ -n "$5" && -f "$csv_file" ]]; then
          temp_file=$(mktemp)
          head -n 1 "$csv_file" > "$temp_file"
          tail -n +2 "$csv_file" | awk -F"," -v suffix="$6" "BEGIN{OFS=\",\"} {\$2=\$2 suffix; print}" >> "$temp_file"
          mv "$temp_file" "$csv_file"
        fi
        ' _ "$1" "$2" "$3" "$pytest_args" "$suffix" "$id_suffix" {} \;
    set -e
}


# Merges all per-file results in a folder (1st arg) into a single CSV.
# NOTE: We currently group the tests such that all files with zero failures
#       come first. To have a more deterministic order, we might consider to
#       change that behavior.
function merge_test_results() {
    # note: status emitted by pytest-csv: passed, failed, error, skipped,
    #       xpassed, xfailed
    # note: status is second last CSV column, so is embedded within ","
    FAILURE_EXPR=",\(error\|failed\),"

    result=""
    test_failures=0
    # first add results with no 'failed' or 'error' status (xfailed are included)
    i=0
    while IFS= read -r f; do
        if (( i == 0 )); then
            result+=$(cat "$f")$'\n'
        else
            result+=$(tail -n +2 "$f")$'\n'
        fi
        i=1
    done < <(grep -rL "${FAILURE_EXPR}" --include="*.csv" "$1")
    # second include results with failed
    while IFS= read -r f; do
        test_failures=1
        if (( i == 0 )); then
            result+=$(cat "$f")$'\n'
        else
            result+=$(tail -n +2 "$f")$'\n'
        fi
        i=1
    done < <(grep -rl "${FAILURE_EXPR}" --include="*.csv" "$1")
    echo "$result"
    return $test_failures
}

# Jenkins ENABLE_CODE_COVERAGE (or HIPVS_ENABLE_CODE_COVERAGE). Do not pass
# --cov into run_tests: that helper is one pytest per file.
coverage_enabled() {
    case "${ENABLE_CODE_COVERAGE:-${HIPVS_ENABLE_CODE_COVERAGE:-0}}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

begin_llvm_profiling() {
    coverage_enabled || return 0
    mkdir -p "$1"
    export LLVM_PROFILE_FILE="$1/%p.profraw"
}

export_llvm_cov_json() {
    coverage_enabled || return 0
    local profdir="$1" json="$2"
    shift 2
    mkdir -p "$(dirname "${json}")"
    shopt -s nullglob
    local raws=("${profdir}"/*.profraw)
    shopt -u nullglob
    if (( ${#raws[@]} == 0 )); then
        echo "WARNING: no LLVM profraw files under ${profdir}; skipping llvm-cov export"
        return 0
    fi
    llvm-profdata merge -sparse "${raws[@]}" -o "${profdir}/merged.profdata"
    local objects=()
    local o
    for o in "$@"; do
        [[ -e "${o}" ]] && objects+=(--object="${o}")
    done
    if (( ${#objects[@]} == 0 )); then
        echo "WARNING: no instrumented objects for llvm-cov export"
        return 0
    fi
    llvm-cov export "${objects[@]}" --Xdemangler=c++filt \
        --instr-profile="${profdir}/merged.profdata" \
        --ignore-filename-regex='_deps' \
        > "${json}"
}

run_pytest_cov() {
    coverage_enabled || return 0
    local pkg="$1" tests="$2"
    local json="${COVERAGE_PYTEST_JSON:-${ARTIFACTS_DIR}/pytest_code_coverage.json}"
    mkdir -p "$(dirname "${json}")"
    python -m pip install -q pytest-cov
    local append=()
    [[ -f "${json}" ]] && append+=(--cov-append)
    pytest -q --cov="${pkg}" --cov-report="json:${json}" --cov-report=term-missing \
        "${append[@]}" "${tests}"
}
