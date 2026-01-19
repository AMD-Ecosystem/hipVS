#!/usr/bin/env python3

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
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

r"""
Benchmark Visualization Site Generator.

Generates an interactive static HTML site from hipVS benchmark results.

Example
-------
Basic usage::

    python generate_site.py gfx90a:/data/result gfx942:/data/result

With system info::

    python generate_site.py gfx90a:/datasets/result gfx942:/datasets/result \
        --system-info gfx90a:/datasets/system_info_gfx90a.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Script directory for locating templates
SCRIPT_DIR = Path(__file__).parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
STATIC_DIR = SCRIPT_DIR / "static"

BUILD_COLUMNS = {"algo_name", "index_name", "time", "GPU"}
SEARCH_COLUMNS = {"algo_name", "index_name", "recall", "throughput", "latency"}


@dataclass
class SystemResult:
    """Benchmark results from a single system."""

    name: str
    path: Path
    build_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    context: dict = field(default_factory=dict)
    system_info: dict = field(default_factory=dict)


def parse_system_arg(arg: str) -> tuple[str, Path]:
    """Parse 'system_name:path' argument."""
    if ":" not in arg:
        # Use directory name as system name
        path = Path(arg)
        name = path.parent.name if path.name == "result" else path.name
        return name, path

    parts = arg.split(":", 1)
    return parts[0], Path(parts[1])


def load_system_info(path: Path) -> dict:
    """Load system info JSON file."""
    if not path.exists():
        print(
            f"  Warning: System info file not found: {path}", file=sys.stderr
        )
        return {}

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(
            f"  Warning: Failed to parse system info JSON: {e}",
            file=sys.stderr,
        )
        return {}


def read_csv_subset(csv_file: Path, needed_cols: set[str]) -> pd.DataFrame:
    """Read a CSV file with only the requested columns if present."""
    try:
        header = pd.read_csv(csv_file, nrows=0)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    usecols = [c for c in header.columns if c in needed_cols]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(csv_file, usecols=usecols)


def load_build_data(result_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load build benchmark data from result directory."""
    build_dir = result_path / "build"
    if not build_dir.exists():
        return pd.DataFrame(), {}

    all_data = []
    context = {}

    for csv_file in build_dir.glob("*.csv"):
        df = read_csv_subset(csv_file, BUILD_COLUMNS)
        if df.empty:
            continue
        algo_name = csv_file.stem.split(",")[0]
        df["algorithm"] = algo_name
        all_data.append(df)

        json_file = csv_file.with_suffix(".json")
        if json_file.exists() and not context:
            data = json.loads(json_file.read_text())
            context = data.get("context", {})

    if all_data:
        return pd.concat(all_data, ignore_index=True), context
    return pd.DataFrame(), context


def load_search_data(result_path: Path) -> pd.DataFrame:
    """Load search benchmark data from result directory."""
    search_dir = result_path / "search"
    if not search_dir.exists():
        return pd.DataFrame()

    all_data = []

    for csv_file in search_dir.glob("*,latency.csv"):
        df = read_csv_subset(csv_file, SEARCH_COLUMNS)
        if df.empty:
            continue
        parts = csv_file.stem.split(",")
        algo_name = parts[0]
        df["algorithm"] = algo_name

        for part in parts:
            if part.startswith("k"):
                df["k_param"] = int(part[1:])
            if part.startswith("bs"):
                df["batch_size"] = int(part[2:])

        all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_system_results(name: str, path: Path) -> SystemResult:
    """Load all benchmark results for a system."""
    result = SystemResult(name=name, path=path)

    if not path.exists():
        print(f"Warning: Path {path} does not exist", file=sys.stderr)
        return result

    result.build_data, result.context = load_build_data(path)
    result.search_data = load_search_data(path)

    if not result.build_data.empty:
        result.build_data["system"] = name
    if not result.search_data.empty:
        result.search_data["system"] = name

    return result


def prepare_chart_data(systems: list[SystemResult]) -> dict:
    """Prepare data for Chart.js visualization.

    Optimized structure to reduce repetition:
    - Single 'series' array instead of duplicate recallLatency/recallThroughput
    - Points stored as compact arrays [recall, latency, throughput, index_idx]
    - Index names in lookup table to avoid repetition
    - Numeric values rounded to reduce precision bloat
    """
    chart_data = {
        "systems": [],
        "algorithms": set(),
        "series": [],
        "indexNames": [],
        "buildTimes": [],
        "systemInfo": {},
    }

    index_name_map = {}

    def get_index_id(name: str) -> int:
        if name not in index_name_map:
            index_name_map[name] = len(chart_data["indexNames"])
            chart_data["indexNames"].append(name)
        return index_name_map[name]

    def round_val(v: float, decimals: int = 6) -> float:
        return round(v, decimals)

    for sys_data in systems:
        if (
            sys_data.search_data.empty
            and sys_data.build_data.empty
            and not sys_data.system_info
            and not sys_data.context
        ):
            continue

        chart_data["systems"].append(
            {
                "name": sys_data.name,
                "gpu": sys_data.context.get("gpu_name", "Unknown GPU"),
                "dataset": sys_data.context.get("dataset", "Unknown"),
                "n_records": sys_data.context.get("n_records", "Unknown"),
            }
        )

        if sys_data.system_info:
            chart_data["systemInfo"][sys_data.name] = sys_data.system_info

        if (
            not sys_data.search_data.empty
            and "algorithm" in sys_data.search_data.columns
        ):
            chart_data["algorithms"].update(
                sys_data.search_data["algorithm"].unique()
            )

        if not sys_data.search_data.empty:
            sorted_search = sys_data.search_data.sort_values(
                ["algorithm", "recall"]
            )

            for algo in sorted_search["algorithm"].unique():
                algo_data = sorted_search[sorted_search["algorithm"] == algo]

                points = [
                    [
                        round_val(float(row.recall), 5),
                        round_val(float(row.latency), 6),
                        round(float(row.throughput)),
                        get_index_id(getattr(row, "index_name", ""))
                        if getattr(row, "index_name", "")
                        else -1,
                    ]
                    for row in algo_data.itertuples(index=False)
                ]

                chart_data["series"].append(
                    {"s": sys_data.name, "a": algo, "p": points}
                )

        if not sys_data.build_data.empty:
            for algo in sys_data.build_data["algorithm"].unique():
                algo_data = sys_data.build_data[
                    sys_data.build_data["algorithm"] == algo
                ]
                for row in algo_data.itertuples(index=False):
                    idx_name = getattr(row, "index_name", "")
                    chart_data["buildTimes"].append(
                        {
                            "s": sys_data.name,
                            "a": algo,
                            "i": get_index_id(idx_name) if idx_name else -1,
                            "t": round_val(float(row.time), 2),
                        }
                    )

    chart_data["algorithms"] = sorted(list(chart_data["algorithms"]))

    # Pre-group build times by index name for the build chart
    # Structure: { indexName: { algo: str, systems: { sysName: time } } }
    build_by_index = {}
    max_time_by_index = {}
    for b in chart_data["buildTimes"]:
        idx_name = chart_data["indexNames"][b["i"]] if b["i"] >= 0 else ""
        if idx_name not in build_by_index:
            build_by_index[idx_name] = {"a": b["a"], "sys": {}}
            max_time_by_index[idx_name] = 0
        build_by_index[idx_name]["sys"][b["s"]] = b["t"]
        if b["t"] > max_time_by_index[idx_name]:
            max_time_by_index[idx_name] = b["t"]

    # Sort indices by max time (descending) and store as list
    sorted_indices = sorted(
        build_by_index.keys(),
        key=lambda x: max_time_by_index.get(x, 0),
        reverse=True,
    )
    chart_data["buildByIndex"] = [
        {
            "n": idx,
            "a": build_by_index[idx]["a"],
            "sys": build_by_index[idx]["sys"],
        }
        for idx in sorted_indices
    ]

    # Pre-compute relative build performance (avg across configs, normalized)
    build_by_algo_system = {}
    for b in chart_data["buildTimes"]:
        key = (b["a"], b["s"])
        if key not in build_by_algo_system:
            build_by_algo_system[key] = []
        build_by_algo_system[key].append(b["t"])

    # Calculate averages
    avg_by_algo = {}
    for (algo, sysname), times in build_by_algo_system.items():
        avg = sum(times) / len(times)
        if algo not in avg_by_algo:
            avg_by_algo[algo] = {}
        avg_by_algo[algo][sysname] = avg

    # Calculate relative performance (fastest = 1.0)
    chart_data["relativeBuild"] = {}
    for algo, sysname_avgs in avg_by_algo.items():
        min_avg = min(sysname_avgs.values())
        chart_data["relativeBuild"][algo] = {
            sn: round(avg / min_avg, 3) for sn, avg in sysname_avgs.items()
        }

    # Pre-compute relative search performance for latency
    # For each algo/system: find optimal point (min latency at max recall)
    optimal_latency_by_algo_system = {}
    for s in chart_data["series"]:
        algo, sys = s["a"], s["s"]
        points = s["p"]  # [recall, latency, throughput, index_idx]
        if not points:
            continue

        # Find max recall achieved
        max_recall = max(p[0] for p in points)
        # Among points at max recall, find min latency
        points_at_max = [p for p in points if p[0] == max_recall]
        best_point = min(points_at_max, key=lambda p: p[1])

        if algo not in optimal_latency_by_algo_system:
            optimal_latency_by_algo_system[algo] = {}
        optimal_latency_by_algo_system[algo][sys] = {
            "recall": round(best_point[0], 5),
            "latency": round(best_point[1], 6),
            "throughput": round(best_point[2]),
        }

    # Pre-compute relative search performance for throughput
    # For each algo/system: find optimal point (max throughput at max recall)
    optimal_throughput_by_algo_system = {}
    for s in chart_data["series"]:
        algo, sys = s["a"], s["s"]
        points = s["p"]  # [recall, latency, throughput, index_idx]
        if not points:
            continue

        # Find max recall achieved
        max_recall = max(p[0] for p in points)
        # Among points at max recall, find max throughput
        points_at_max = [p for p in points if p[0] == max_recall]
        best_point = max(points_at_max, key=lambda p: p[2])

        if algo not in optimal_throughput_by_algo_system:
            optimal_throughput_by_algo_system[algo] = {}
        optimal_throughput_by_algo_system[algo][sys] = {
            "recall": round(best_point[0], 5),
            "latency": round(best_point[1], 6),
            "throughput": round(best_point[2]),
        }

    # Calculate relative latency (fastest = 1.0, higher = slower)
    chart_data["relativeSearchLatency"] = {}
    for algo, sys_data in optimal_latency_by_algo_system.items():
        min_latency = min(d["latency"] for d in sys_data.values())
        chart_data["relativeSearchLatency"][algo] = {
            sys: {
                "relative": round(data["latency"] / min_latency, 3),
                "recall": data["recall"],
                "latency": data["latency"],
                "throughput": data["throughput"],
            }
            for sys, data in sys_data.items()
        }

    # Calculate relative throughput (slowest = 1.0, higher = faster)
    # Higher throughput is better, so faster systems get higher multipliers
    chart_data["relativeSearchThroughput"] = {}
    for algo, sys_data in optimal_throughput_by_algo_system.items():
        min_throughput = min(d["throughput"] for d in sys_data.values())
        chart_data["relativeSearchThroughput"][algo] = {
            sys: {
                "relative": round(data["throughput"] / min_throughput, 3)
                if min_throughput > 0
                else 0,
                "recall": data["recall"],
                "latency": data["latency"],
                "throughput": data["throughput"],
            }
            for sys, data in sys_data.items()
        }

    return chart_data


def load_template() -> str:
    """Load the HTML template file."""
    template_path = TEMPLATES_DIR / "index.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text()


def load_css() -> str:
    """Load the CSS file."""
    css_path = STATIC_DIR / "style.css"
    if not css_path.exists():
        raise FileNotFoundError(f"CSS not found: {css_path}")
    return css_path.read_text()


def generate_html(chart_data: dict, output_path: Path):
    """Generate the static HTML visualization site."""
    # Load template and CSS
    template = load_template()
    css = load_css()

    # Replace placeholders
    html_content = template.replace("{{INLINE_CSS}}", css)
    html_content = html_content.replace(
        "{{CHART_DATA}}", json.dumps(chart_data, separators=(",", ":"))
    )

    # Write the HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    print(f"Generated visualization site: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="""
Generate an interactive HTML visualization site from hipVS benchmark results.

Compare benchmark performance across multiple GPU systems with interactive
charts for recall vs latency, recall vs throughput, and build times.
""".strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Format:
  Systems can be specified as NAME:PATH or just PATH.
  - NAME:PATH   Use NAME as the system label (e.g., "gfx90a:/data/result")
  - PATH        Derive name from directory (e.g., "/data/gfx90a/result")

  The PATH should point to a benchmark result directory containing:
    result/
    ├── build/     # Build benchmark CSVs (*.csv) and metadata (*.json)
    └── search/    # Search benchmark CSVs (*,latency.csv)

Examples:
  # Compare two GPU systems
  %(prog)s gfx90a:/datasets/result gfx942:/datasets/result

  # Specify output directory (creates DIR/index.html)
  %(prog)s /data/gfx90a/result /data/gfx942/result -o ./comparison

  # Specify output file directly
  %(prog)s /data/gfx90a/result /data/gfx942/result -o ./report.html

  # Include detailed system hardware info
  %(prog)s gfx90a:/data/result gfx942:/data/result \\
      --system-info gfx90a:system_info_gfx90a.json \\
                    gfx942:system_info_gfx942.json

Output:
  Generates a single HTML file with embedded CSS and data.
  - If -o ends with .html: writes to that file directly
  - Otherwise: writes to PATH/index.html
  Open in a browser or serve with: python -m http.server -d OUTPUT_DIR
""",
    )

    parser.add_argument(
        "systems",
        nargs="+",
        metavar="SYSTEM",
        help="benchmark results as NAME:PATH or PATH (at least one required)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="PATH",
        default=Path("./benchmark_site"),
        help="output directory or .html file (default: ./benchmark_site/)",
    )

    parser.add_argument(
        "--system-info",
        nargs="*",
        metavar="NAME:PATH",
        dest="system_info",
        help="system hardware info JSON files (NAME must match a SYSTEM name)",
    )

    args = parser.parse_args()

    # Parse system info arguments into a dict
    system_info_map = {}
    if args.system_info:
        for info_arg in args.system_info:
            name, path = parse_system_arg(info_arg)
            system_info_map[name] = path

    # Load all system results
    systems = []
    for sys_arg in args.systems:
        name, path = parse_system_arg(sys_arg)
        print(f"Loading results for {name} from {path}...")
        result = load_system_results(name, path)

        if result.search_data.empty:
            print(f"  Warning: No search data found for {name}")
        else:
            print(f"  Loaded {len(result.search_data)} search results")

        if result.build_data.empty:
            print(f"  Warning: No build data found for {name}")
        else:
            print(f"  Loaded {len(result.build_data)} build results")

        if name in system_info_map:
            print(f"  Loading system info from {system_info_map[name]}...")
            result.system_info = load_system_info(system_info_map[name])
            if result.system_info:
                coll_info = result.system_info.get("collection_info", {})
                hostname = coll_info.get("hostname", name)
                print(f"  Loaded system info for {hostname}")

        systems.append(result)

    # Warn about system info files that don't match any system name
    if system_info_map:
        known_systems = {s.name for s in systems}
        unknown_infos = sorted(set(system_info_map.keys()) - known_systems)
        if unknown_infos:
            unknown_str = ", ".join(unknown_infos)
            print(
                f"Warning: System info for unknown system(s): {unknown_str}",
                file=sys.stderr,
            )

    # Prepare chart data
    print("\nPreparing visualization data...")
    chart_data = prepare_chart_data(systems)

    # Generate HTML - if path ends with .html, use directly; else treat as dir
    if args.output.suffix.lower() == ".html":
        output_file = args.output
        output_dir = args.output.parent
    else:
        output_file = args.output / "index.html"
        output_dir = args.output

    generate_html(chart_data, output_file)

    print(
        f"\nDone! Open {output_file} in a browser to view the visualization."
    )
    print(f"Tip: Use 'python -m http.server -d {output_dir}' to serve locally")


if __name__ == "__main__":
    main()
