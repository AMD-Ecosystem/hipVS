#!/usr/bin/env bash
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#
# Regenerate pybind11-stubgen type stubs for the cuvs (hipVS) Python package.
# Usage: ./generate_stubs.sh
#
# Prerequisites: a Python environment with both `cuvs` and `pybind11-stubgen`
# installed. The script will exit early if either is missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUBS_DIR="$SCRIPT_DIR/stubs/cuvs"

python -c "import cuvs" 2>/dev/null || { echo "Error: cuvs is not installed in the current Python environment." >&2; exit 1; }
command -v pybind11-stubgen >/dev/null 2>&1 || { echo "Error: pybind11-stubgen is not installed. Install with: pip install pybind11-stubgen" >&2; exit 1; }

readarray -t MODULES < <(python -c "
import pkgutil, cuvs
print('cuvs')
for _, name, _ in pkgutil.walk_packages(cuvs.__path__, prefix='cuvs.'):
    print(name)
")

LICENSE_HEADER='# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
'

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Generating stubs into $TMPDIR ..."
for mod in "${MODULES[@]}"; do
    echo "  $mod"
    pybind11-stubgen --ignore-all-errors -o "$TMPDIR" "$mod" 2>/dev/null || true
done

GENERATED=$(find "$TMPDIR/cuvs" -name "*.pyi" 2>/dev/null | wc -l)
if [ "$GENERATED" -eq 0 ]; then
    echo "Error: no stub files were generated." >&2
    exit 1
fi

echo "Replacing $STUBS_DIR ..."
rm -rf "$STUBS_DIR"
cp -r "$TMPDIR/cuvs" "$STUBS_DIR"

echo "Prepending license headers ..."
find "$STUBS_DIR" -name "*.pyi" | while read -r f; do
    tmp=$(mktemp)
    printf '%s\n' "$LICENSE_HEADER" | cat - "$f" > "$tmp"
    mv "$tmp" "$f"
done

echo "Done. Generated $GENERATED stub files."
