#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

dtype_sizes = {
    "float": 4,
    "fp8": 1,
    "half": 2,
}


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

    # Constrain search_width * graph_degree (== num_itopk_candidates) to keep the
    # single-CTA search kernel's dynamic shared memory within the device LDS limit.
    #
    # The dynamic smem requirement (see search_single_cta.cuh::set_params) scales with:
    #   result_buffer_size = itopk_size + search_width * graph_degree
    # and, once num_itopk_candidates > 256, also includes the radix-sort topk buffer.
    # On AMD GPUs the per-block LDS is 64 KiB (gfx90a: sharedMemPerBlock == 65536).
    # At num_itopk_candidates == 2048 the result buffer + radix buffer + dataset
    # workspace exceed 64 KiB and cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
    # fails at launch. Empirically num_itopk_candidates <= 1024 stays within budget
    # across itopk in {128, 256, 512}, while 2048 overflows.
    #
    # (1024 also keeps block_size within max_block_size (1024) for the radix path.)
    if "search_width" in params and "graph_degree" in build_params:
        num_itopk_candidates = (
            params["search_width"] * build_params["graph_degree"]
        )
        if num_itopk_candidates > 1024:
            return False

    return True


###############################################################################
#                              FAISS constraints                              #
###############################################################################


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
