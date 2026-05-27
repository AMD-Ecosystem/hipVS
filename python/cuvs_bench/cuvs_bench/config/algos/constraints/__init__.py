#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# MIT License
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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


###############################################################################
#                                 Utilities                                   #
###############################################################################

import functools
import warnings

dtype_sizes = {
    "float": 4,
    "fp8": 1,
    "half": 2,
}

DEFAULT_SMEM = 49152  # 48 KiB fallback when hip-python is unavailable
# Empirical gfx90a constants (see search_single_cta.cuh::set_params, topk_by_radix.cuh):
# - dataset descriptor smem overhead: ~436 B measured, rounded up to 512 B for safety
# - kernel static smem footprint: 15-17 KiB inferred from failure point; 16 KiB used
_DATASET_DESC_OVERHEAD = 512
_STATIC_SMEM_ESTIMATE = 16384


@functools.lru_cache(maxsize=None)
def _smem_per_block(device: int = 0) -> int:
    try:
        from hip import hip as hipp
        err, props = hipp.hipGetDeviceProperties(device)
        if err == 0:
            return props.sharedMemPerBlock
    except Exception:
        pass
    return DEFAULT_SMEM


@functools.lru_cache(maxsize=None)
def _cagra_dynamic_smem_limit(device: int = 0) -> int:
    # Deduct kernel static smem; cannot call hipFuncGetAttribute without kernel pointer.
    return max(0, _smem_per_block(device) - _STATIC_SMEM_ESTIMATE)


def _cagra_single_cta_smem(itopk: int, search_width: int, graph_degree: int) -> int:
    # Mirrors search_single_cta.cuh::set_params() — INDEX_T=uint32, DISTANCE_T=float.
    # Assumes no sample filter and small_hash_bitlen=0 (default search path).
    num_itopk_candidates = search_width * graph_degree
    result_buffer_size_64 = (itopk + num_itopk_candidates + 63) & ~63
    base_smem = _DATASET_DESC_OVERHEAD + 8 * result_buffer_size_64 + 4 * search_width + 16
    if num_itopk_candidates > 256:
        max_itopk = 256 if itopk <= 256 else 512
        return base_smem + (max_itopk * 2 + 2048 + 8) * 4
    return base_smem


###############################################################################
#                              cuVS constraints                               #
###############################################################################


def cuvs_cagra_build(params, dims):
    if "graph_degree" in params and "intermediate_graph_degree" in params:
        return params["graph_degree"] <= params["intermediate_graph_degree"]
    return True


def cuvs_ivf_pq_build(params, dims):
    if "pq_dim" in params:
        return params["pq_dim"] <= dims
    return True


def cuvs_ivf_pq_search(params, build_params, k, batch_size):
    ret = True
    if "internalDistanceDtype" in params and "smemLutDtype" in params:
        ret = (
            dtype_sizes[params["smemLutDtype"]]
            <= dtype_sizes[params["internalDistanceDtype"]]
        )

    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


def cuvs_cagra_search(params, build_params, k, batch_size):
    if "itopk" in params:
        if params["itopk"] < k:
            return False
        # itopk_size must not exceed 512 (max_itopk in search_single_cta.cuh)
        if params["itopk"] > 512:
            return False

    # Reject configs whose estimated single-CTA kernel smem exceeds the device LDS limit.
    # _cagra_single_cta_smem mirrors search_single_cta.cuh::set_params(); sharedMemPerBlock
    # is queried at runtime so the constraint adapts to the device automatically.
    if "search_width" in params and "graph_degree" in build_params and "itopk" in params:
        itopk = params["itopk"]
        search_width = params["search_width"]
        graph_degree = build_params["graph_degree"]
        estimated_smem = _cagra_single_cta_smem(itopk, search_width, graph_degree)
        smem_limit = _cagra_dynamic_smem_limit()
        if estimated_smem > smem_limit:
            warnings.warn(
                f"CAGRA search config dropped: estimated smem {estimated_smem} B "
                f"> limit {smem_limit} B "
                f"(itopk={itopk}, search_width={search_width}, graph_degree={graph_degree}). "
                f"Reduce search_width or graph_degree.",
                stacklevel=2,
            )
            return False

    return True


def cuvs_ivf_sq_search(params, build_params, k, batch_size):
    if "nlist" in build_params and "nprobe" in params:
        return build_params["nlist"] >= params["nprobe"]
    return True


###############################################################################
#                              FAISS constraints                              #
###############################################################################


def faiss_gpu_ivf_sq_search(params, build_params, k, batch_size):
    if "nlist" in build_params and "nprobe" in params:
        return build_params["nlist"] >= params["nprobe"]
    return True


def faiss_cpu_ivf_sq_search(params, build_params, k, batch_size):
    if "nlist" in build_params and "nprobe" in params:
        return build_params["nlist"] >= params["nprobe"]
    return True


def faiss_gpu_ivf_pq_build(params, dims):
    ret = True
    # M must be defined
    ret = params["M"] <= dims and dims % params["M"] == 0
    if "use_cuvs" in params and params["use_cuvs"]:
        return ret
    pq_bits = params.get("bitsPerCode", 8)
    lookup_table_size = 4
    if "useFloat16" in params and params["useFloat16"]:
        lookup_table_size = 2
    # FAISS constraint to check if lookup table fits in shared memory
    # for now hard code maximum shared memory per block to 49 kB
    # (the value for A100 and V100)
    return ret and lookup_table_size * params["M"] * (2**pq_bits) <= 49152


def faiss_gpu_ivf_pq_search(params, build_params, k, batch_size):
    ret = True
    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


###############################################################################
#                              hnswlib constraints                            #
###############################################################################


def hnswlib_search(params, build_params, k, batch_size):
    if "ef" in params:
        return params["ef"] >= k


###############################################################################
#                              DiskANN constraints                            #
###############################################################################


def diskann_memory_build(params, dim):
    ret = True
    if "R" in params and "L_build" in params:
        ret = params["R"] <= params["L_build"]
    return ret


def diskann_ssd_build(params, dim):
    ret = True
    if "R" in params and "L_build" in params:
        ret = params["R"] <= params["L_build"]
    if "QD" in params:
        ret = params["QD"] <= dim
    return ret
