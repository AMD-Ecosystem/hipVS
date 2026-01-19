<!---
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
-->

# Benchmark Report Generator

Generate interactive HTML visualization sites from hipVS benchmark results.

Compare benchmark performance across multiple GPU systems with interactive charts for:
- Recall vs Latency
- Recall vs Throughput
- Build Times

## Quick Start

```bash
# Compare two GPU systems
python generate_html_report.py gfx90a:/datasets/result gfx942:/datasets/result

# Open the generated site
open benchmark_site/index.html
```

## Usage

```
generate_html_report.py [-h] [-o PATH] [--system-info [NAME:PATH ...]] SYSTEM [SYSTEM ...]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `SYSTEM` | Benchmark results as `NAME:PATH` or `PATH` (at least one required) |
| `-o PATH, --output PATH` | Output directory or `.html` file (default: `./benchmark_site/`) |
| `--system-info [NAME:PATH ...]` | System hardware info JSON files (NAME must match a SYSTEM name) |

## Input Format

Systems can be specified as `NAME:PATH` or just `PATH`:

| Format | Description | Example |
|--------|-------------|---------|
| `NAME:PATH` | Use NAME as the system label | `gfx90a:/data/result` |
| `PATH` | Derive name from directory | `/data/gfx90a/result` → `gfx90a` |

### Expected Directory Structure

The `PATH` should point to a benchmark result directory containing:

```
result/
├── build/     # Build benchmark CSVs (*.csv) and metadata (*.json)
└── search/    # Search benchmark CSVs (*,latency.csv)
```

## Output

Generates a single HTML file with embedded CSS and data:
- If `-o` ends with `.html`: writes to that file directly
- Otherwise: writes to `PATH/index.html`

View in a browser directly or serve locally:

```bash
python -m http.server -d ./benchmark_site
```

## Examples

### Basic comparison of two GPUs

```bash
python generate_html_report.py \
    gfx90a:/datasets/deep-image-96-inner/result \
    gfx942:/datasets/gfx942/result
```

### Specify output directory

```bash
python generate_html_report.py \
    /data/gfx90a/result \
    /data/gfx942/result \
    -o ./comparison
# Creates: ./comparison/index.html
```

### Specify output file directly

```bash
python generate_html_report.py \
    /data/gfx90a/result \
    /data/gfx942/result \
    -o ./my_report.html
# Creates: ./my_report.html
```

### Multi-system comparison with hardware info

```bash
python generate_html_report.py \
    gfx90a:/datasets/deep-image-96-inner/result \
    gfx942:/datasets/gfx942/result \
    h100:/datasets/h100/result/ \
    gfx950:/datasets/gfx950/result/ \
    --system-info \
        gfx90a:/datasets/system_info_gfx90a.json \
        gfx942:/datasets/system_info_gfx942.json \
        h100:/datasets/system_info_h100.json \
        gfx950:/datasets/system_info_gfx950.json \
    -o benchmark_site/index.html
```

## Collecting System Info

Use the `collect_system_info.sh` script to gather hardware information:

```bash
./collect_system_info.sh
# Outputs: system_info_<hostname>_<gpu>_<timestamp>.json
```

This JSON file can be passed via `--system-info` to include hardware details in the generated report.

## Files

| File | Description |
|------|-------------|
| `generate_html_report.py` | Main script |
| `collect_system_info.sh` | Helper to collect system hardware info |
| `templates/index.html` | HTML template with Chart.js visualization |
| `static/style.css` | Stylesheet for the generated site |
