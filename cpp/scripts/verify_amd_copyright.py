#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import subprocess
from enum import Enum
import re
from datetime import datetime
from pathlib import Path


# To be updated every time we pull in upstream changes.
BASE_COMMIT_HASH = "2b7dce65abf8334664f00c7bb23076caaf603f6e"

CURRENT_YEAR = str(datetime.now().year)
BASE_PATTERN = "Advanced Micro Devices"
PATTERN = re.compile(
    rf"(?P<modifications_amd>Modifications Copyright.*{BASE_PATTERN})|"
    rf"(?P<copyright_amd>Copyright.*{BASE_PATTERN})"
)

FILES_ADDED_IN_NEW_UPSTREAM_COMMITS = {
    "cpp/include/cuvs/preprocessing/quantize/binary.hpp",
    "cpp/tests/preprocessing/binary_quantization.cu",
    "cpp/src/preprocessing/quantize/detail/binary.cuh"
}


class FileStatus(Enum):
    ADDED = 1
    MODIFIED = 2
    OTHER = 3


def get_change_status(file_path, base_commit) -> FileStatus:
    try:
        diff_output = subprocess.check_output(
            ["git", "diff", "--name-status", base_commit, "--", file_path],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        # If something goes wrong, handle or raise an exception as needed.
        print(f"Error running git diff on {file_path}: {e}", file=sys.stderr)
        exit(1)

    if not diff_output:
        # If there's no output, the file is unchanged relative to base_commit
        return FileStatus.OTHER

    # Parse the status line. For a single file, it's typically one line:
    # e.g., "M <filename>" or "A <filename>"
    line = diff_output.splitlines()[0]
    status_code = line.split("\t")[0]

    if status_code == "A":
        return FileStatus.ADDED
    elif status_code == "M":
        return FileStatus.MODIFIED
    elif status_code == "R":
        return FileStatus.MODIFIED
    else:
        return FileStatus.OTHER


def validate_modified_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    match = PATTERN.search(content)
    if match is None:
        print(f"[ERROR] AMD Modifications Copyright is missing from {file_path}")
    else:
        match_dict = match.groupdict()
        if match_dict["modifications_amd"] is None:
            print(
                f"[ERROR] Modifications AMD Copyright not found in {file_path} that's being modified",
                file=sys.stderr,
            )
        elif match_dict["copyright_amd"] is not None:
            print(
                f"[ERROR] AMD Copyright found in {file_path} that's being modified. Add the Modifications Copyright instead.",
                file=sys.stderr,
            )
        elif CURRENT_YEAR not in match_dict["modifications_amd"]:
            print(
                f"[ERROR] {file_path} does not have the current year in the modifications copyright",
                file=sys.stderr,
            )
        else:
            # All checks have passed
            return True

    return False


def validate_new_file(file_path):
    if str(file_path) in FILES_ADDED_IN_NEW_UPSTREAM_COMMITS:
        return True
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    match = PATTERN.search(content)
    if match is None:
        print(f"[ERROR] AMD copyright is missing from {file_path}")
    else:
        match_dict = match.groupdict()
        if match_dict["modifications_amd"] is not None:
            print(
                f"[ERROR] Found Modifications Copyright in {file_path} that's being added. Add the \"MIT License\" instead.",
                file=sys.stderr,
            )
        elif match_dict["copyright_amd"] is None:
            print(
                f"[ERROR] AMD Copyright not found in {file_path} that's being added",
                file=sys.stderr,
            )
        elif CURRENT_YEAR not in match_dict["copyright_amd"]:
            print(
                f"[ERROR] AMD Copyright {file_path} does not have the current year",
                file=sys.stderr,
            )
        else:
            # All checks have passed
            return True
    return False


if __name__ == "__main__":
    staged_files = [Path(file) for file in sys.argv[1:]]
    validation_results = list()
    for f in staged_files:
        change_type = get_change_status(f, BASE_COMMIT_HASH)
        if change_type == FileStatus.ADDED:
            validation_results.append(validate_new_file(f))
        elif change_type == FileStatus.MODIFIED:
            validation_results.append(validate_modified_file(f))
        elif change_type == FileStatus.OTHER:
            continue
        else:
            exit(1)
    if all(validation_results):
        exit(0)
    else:
        print(f"[ERROR] Copyright validation failed", file=sys.stderr)
        exit(1)
