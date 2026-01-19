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

#!/bin/bash
#
# System Information Collection Script for GPU Benchmarking
# Collects hardware specs, software versions, and configuration for reproducibility
#
# Usage: ./collect_system_info.sh [output_dir]
#
set -e
# Output directory (default: current directory)
OUTPUT_DIR="${1:-.}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname -s)
# Detect GPU architecture
GPU_ARCH="unknown"
if command -v rocminfo &> /dev/null; then
    GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "gfx" | grep -oE "gfx[0-9a-z]+" || echo "unknown")
fi
OUTPUT_FILE="${OUTPUT_DIR}/system_info_${HOSTNAME}_${GPU_ARCH}_${TIMESTAMP}.txt"
JSON_FILE="${OUTPUT_DIR}/system_info_${HOSTNAME}_${GPU_ARCH}_${TIMESTAMP}.json"
echo "=============================================="
echo "  System Information Collection Script"
echo "=============================================="
echo "Output: ${OUTPUT_FILE}"
echo ""
# Start collecting
{
    echo "========================================================================"
    echo "  SYSTEM INFORMATION REPORT"
    echo "  Generated: $(date)"
    echo "  Hostname: $(hostname)"
    echo "========================================================================"
    echo ""
    # -------------------------------------------------------------------------
    echo "=== BASIC SYSTEM INFO ==="
    echo "Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "Uptime: $(uptime)"
    echo "Kernel: $(uname -r)"
    echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo "Unknown")"
    echo "Architecture: $(uname -m)"
    echo ""
    # -------------------------------------------------------------------------
    echo "=== CPU INFORMATION ==="
    if [ -f /proc/cpuinfo ]; then
        echo "Model: $(grep -m1 'model name' /proc/cpuinfo | cut -d':' -f2 | xargs)"
        echo "Physical CPUs: $(grep 'physical id' /proc/cpuinfo | sort -u | wc -l)"
        echo "Total Cores: $(grep -c 'processor' /proc/cpuinfo)"
        echo "Cores per Socket: $(grep -m1 'cpu cores' /proc/cpuinfo | cut -d':' -f2 | xargs)"
        echo "Threads per Core: $(grep -m1 'siblings' /proc/cpuinfo | cut -d':' -f2 | xargs)"
        echo "CPU MHz (current): $(grep -m1 'cpu MHz' /proc/cpuinfo | cut -d':' -f2 | xargs)"
        echo "Cache Size: $(grep -m1 'cache size' /proc/cpuinfo | cut -d':' -f2 | xargs)"
        echo "CPU Flags: $(grep -m1 'flags' /proc/cpuinfo | cut -d':' -f2 | xargs | tr ' ' '\n' | grep -E 'avx|sse|fma' | tr '\n' ' ')"
    fi
    if command -v lscpu &> /dev/null; then
        echo ""
        echo "--- lscpu summary ---"
        lscpu | grep -E "^(Architecture|CPU\(s\)|Thread|Core|Socket|Model name|CPU max MHz|CPU min MHz|L1|L2|L3|NUMA)"
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== MEMORY INFORMATION ==="
    if [ -f /proc/meminfo ]; then
        echo "Total RAM: $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
        echo "Free RAM: $(grep MemFree /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
        echo "Available RAM: $(grep MemAvailable /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
        echo "Swap Total: $(grep SwapTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
        echo "Huge Pages Total: $(grep HugePages_Total /proc/meminfo | awk '{print $2}')"
        echo "Huge Page Size: $(grep Hugepagesize /proc/meminfo | awk '{print $2, $3}')"
    fi
    if command -v dmidecode &> /dev/null && [ "$EUID" -eq 0 ]; then
        echo ""
        echo "--- Memory DIMMs ---"
        dmidecode -t memory 2>/dev/null | grep -E "Size:|Speed:|Type:|Manufacturer:" | head -20
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== GPU INFORMATION ==="

    # ROCm GPU info
    if command -v rocminfo &> /dev/null; then
        echo "--- rocminfo ---"
        rocminfo 2>/dev/null | grep -E "Marketing Name|Name:|Compute Unit|Max Clock|Max Memory|L2 cache|Memory bus|Agent|Vendor" | head -50
        echo ""
    fi

    if command -v rocm-smi &> /dev/null; then
        echo "--- rocm-smi ---"
        rocm-smi --showproductname 2>/dev/null || true
        echo ""
        rocm-smi --showmeminfo vram 2>/dev/null || true
        echo ""
        rocm-smi --showclocks 2>/dev/null || true
        echo ""
        rocm-smi --showmemvendor 2>/dev/null || true
        echo ""
        rocm-smi --showdriverversion 2>/dev/null || true
        echo ""
        echo "--- Full rocm-smi output ---"
        rocm-smi 2>/dev/null || true
    fi

    # HIP info
    if command -v hipconfig &> /dev/null; then
        echo ""
        echo "--- hipconfig ---"
        hipconfig --full 2>/dev/null || hipconfig 2>/dev/null || true
    fi

    # hipcc version
    if command -v hipcc &> /dev/null; then
        echo ""
        echo "--- hipcc version ---"
        hipcc --version 2>/dev/null || true
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== ROCm SOFTWARE VERSIONS ==="

    # ROCm version
    if [ -f /opt/rocm/.info/version ]; then
        echo "ROCm Version: $(cat /opt/rocm/.info/version)"
    elif [ -f /opt/rocm/version ]; then
        echo "ROCm Version: $(cat /opt/rocm/version)"
    fi

    # Check ROCm packages
    if command -v dpkg &> /dev/null; then
        echo ""
        echo "--- ROCm packages (dpkg) ---"
        dpkg -l 2>/dev/null | grep -i rocm | awk '{print $2, $3}' | head -20
    elif command -v rpm &> /dev/null; then
        echo ""
        echo "--- ROCm packages (rpm) ---"
        rpm -qa 2>/dev/null | grep -i rocm | head -20
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== PYTHON & CONDA/MAMBA ENVIRONMENT ==="

    echo "Python: $(which python 2>/dev/null || echo 'not found')"
    python --version 2>/dev/null || true

    echo ""
    echo "Micromamba: $(which micromamba 2>/dev/null || echo 'not found')"
    micromamba --version 2>/dev/null || true

    echo ""
    echo "Mamba: $(which mamba 2>/dev/null || echo 'not found')"
    mamba --version 2>/dev/null | head -1 || true

    echo ""
    echo "Conda: $(which conda 2>/dev/null || echo 'not found')"
    conda --version 2>/dev/null || true

    echo ""
    echo "Active environment: ${CONDA_PREFIX:-${MAMBA_ROOT_PREFIX:-none}}"

    if [ -n "$CONDA_PREFIX" ]; then
        echo ""
        echo "--- Key packages in environment ---"
        # Try micromamba first, then mamba, then conda
        if command -v micromamba &> /dev/null; then
            micromamba list 2>/dev/null | grep -iE "cuvs|hip|rocm|numpy|pytorch|torch|cuda|raft" || true
        elif command -v mamba &> /dev/null; then
            mamba list 2>/dev/null | grep -iE "cuvs|hip|rocm|numpy|pytorch|torch|cuda|raft" || true
        else
            conda list 2>/dev/null | grep -iE "cuvs|hip|rocm|numpy|pytorch|torch|cuda|raft" || true
        fi
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== STORAGE INFORMATION ==="
    echo "--- Disk usage ---"
    df -h 2>/dev/null | grep -E "^/dev|Filesystem"
    echo ""

    if command -v lsblk &> /dev/null; then
        echo "--- Block devices ---"
        lsblk -d -o NAME,SIZE,TYPE,TRAN,MODEL 2>/dev/null | head -15
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== NETWORK INFORMATION ==="
    echo "--- Network interfaces ---"
    ip addr show 2>/dev/null | grep -E "^[0-9]:|inet " | head -20 || ifconfig 2>/dev/null | head -30
    echo ""

    if command -v ibstat &> /dev/null; then
        echo "--- InfiniBand status ---"
        ibstat 2>/dev/null | head -30 || true
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== PCIe TOPOLOGY ==="
    if command -v lspci &> /dev/null; then
        echo "--- GPUs in lspci ---"
        lspci 2>/dev/null | grep -iE "VGA|3D|Display|AMD.*Instinct" || true
        echo ""
    fi

    if command -v rocm-smi &> /dev/null; then
        echo "--- GPU topology ---"
        rocm-smi --showtopo 2>/dev/null || true
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "=== ENVIRONMENT VARIABLES ==="
    echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
    echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-not set}"
    echo "GPU_DEVICE_ORDINAL: ${GPU_DEVICE_ORDINAL:-not set}"
    echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-not set}"
    echo "RAPIDS_DATASET_ROOT_DIR: ${RAPIDS_DATASET_ROOT_DIR:-not set}"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
    echo "PATH: ${PATH}"
    echo ""
    # -------------------------------------------------------------------------
    echo "=== BENCHMARK EXECUTABLE INFO ==="

    # Check for cuVS benchmark executables
    for exe in CUVS_CAGRA_ANN_BENCH CUVS_IVF_FLAT_ANN_BENCH CUVS_IVF_PQ_ANN_BENCH; do
        exe_path=$(which $exe 2>/dev/null || find ~/.local -name "$exe" -type f 2>/dev/null | head -1)
        if [ -n "$exe_path" ]; then
            echo "$exe: $exe_path"
            file "$exe_path" 2>/dev/null || true
        fi
    done
    echo ""
    # -------------------------------------------------------------------------
    echo "=== NUMA TOPOLOGY ==="
    if command -v numactl &> /dev/null; then
        numactl --hardware 2>/dev/null || true
    fi
    echo ""
    # -------------------------------------------------------------------------
    echo "========================================================================"
    echo "  END OF REPORT"
    echo "========================================================================"
} | tee "${OUTPUT_FILE}"
# -------------------------------------------------------------------------
# Also create a JSON summary for programmatic access
# -------------------------------------------------------------------------
echo ""
echo "Creating JSON summary..."
cat > "${JSON_FILE}" << JSONEOF
{
  "collection_info": {
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "script_version": "1.0"
  },
  "system": {
    "os": "$(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo "Unknown")",
    "kernel": "$(uname -r)",
    "architecture": "$(uname -m)"
  },
  "cpu": {
    "model": "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d':' -f2 | xargs || echo "Unknown")",
    "physical_cpus": $(grep 'physical id' /proc/cpuinfo 2>/dev/null | sort -u | wc -l || echo 0),
    "total_cores": $(grep -c 'processor' /proc/cpuinfo 2>/dev/null || echo 0),
    "cores_per_socket": $(grep -m1 'cpu cores' /proc/cpuinfo 2>/dev/null | cut -d':' -f2 | xargs || echo 0)
  },
  "memory": {
    "total_gb": $(grep MemTotal /proc/meminfo 2>/dev/null | awk '{printf "%.1f", $2/1024/1024}' || echo 0),
    "swap_gb": $(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{printf "%.1f", $2/1024/1024}' || echo 0)
  },
  "gpu": {
    "arch": "${GPU_ARCH}",
    "name": "$(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | cut -d':' -f2 | xargs || echo "Unknown")",
    "count": $(rocm-smi --showproductname 2>/dev/null | grep -c "GPU" || echo 0),
    "vram_gb": "$(rocm-smi --showmeminfo vram 2>/dev/null | grep -m1 'Total' | awk '{print $3}' || echo "Unknown")",
    "driver_version": "$(rocm-smi --showdriverversion 2>/dev/null | grep -m1 'Driver' | awk '{print $NF}' || echo "Unknown")"
  },
  "rocm": {
    "version": "$(cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/version 2>/dev/null || echo "Unknown")"
  },
  "environment": {
    "conda_prefix": "${CONDA_PREFIX:-null}",
    "hip_visible_devices": "${HIP_VISIBLE_DEVICES:-null}",
    "python_version": "$(python --version 2>&1 | awk '{print $2}' || echo "Unknown")"
  }
}
JSONEOF
echo ""
echo "=============================================="
echo "  Collection Complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  Text report: ${OUTPUT_FILE}"
echo "  JSON summary: ${JSON_FILE}"
echo ""
echo "To copy these files, run:"
echo "  scp ${OUTPUT_FILE} ${JSON_FILE} user@destination:path/"
echo ""
