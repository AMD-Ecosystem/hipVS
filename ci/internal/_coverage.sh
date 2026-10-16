#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

coverage_enabled() {
  case "${ENABLE_CODE_COVERAGE:-false}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

# Print the normal whitespace-separated matrix, or only 3.12 for coverage.
coverage_python_versions() {
  if coverage_enabled; then
    echo "3.12"
  else
    echo "$1"
  fi
}
