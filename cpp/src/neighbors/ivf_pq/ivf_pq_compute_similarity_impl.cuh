/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once

#include "../ivf_common.cuh"     // dummy_block_sort_t
#include "../sample_filter.cuh"  // none_sample_filter
#include "detail/jit_lto_kernels/block_sort.cuh"
#include "detail/jit_lto_kernels/compute_similarity_planner.hpp"
#include "detail/jit_lto_kernels/kernel_def.cuh"
#include "ivf_pq_compute_similarity.hpp"  // cuvs::neighbors::ivf_pq::detail::selected
#include "ivf_pq_fp_8bit.cuh"
#include <cuvs/distance/distance.hpp>  // cuvs::distance::DistanceType
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>               // codebook_gen
#include <raft/matrix/detail/select_warpsort.cuh>  // matrix::detail::select::warpsort::warp_sort_distributed
#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY
#include <raft/util/device_atomics.cuh>      // raft::atomicMin
#include <raft/util/pow2_utils.cuh>          // raft::Pow2
#include <raft/util/vectorized.cuh>          // raft::TxN_t
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#endif
#include <algorithm>
#include <rmm/cuda_stream_view.hpp>  // rmm::cuda_stream_view
#include <type_traits>

namespace cuvs::neighbors::ivf_pq::detail {

/**
 * Maximum value of k for the fused calculate & select in ivfpq.
 *
 * If runtime value of k is larger than this, the main search operation
 * is split into two kernels (per batch, first calculate distance, then select top-k).
 */
static constexpr int kMaxCapacity = 128;
static_assert((kMaxCapacity >= 32) && !(kMaxCapacity & (kMaxCapacity - 1)),
              "kMaxCapacity must be a power of two, not smaller than the WarpSize.");

// using weak attribute here, because it may be compiled multiple times.
auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries)
  -> bool
{
  if (k > kMaxCapacity) { return false; }            // warp_sort not possible
  if (n_queries * n_probes <= 16) { return false; }  // overall amount of work is too small
  return true;
}

/**
 * Estimate a carveout value as expected by `cudaFuncAttributePreferredSharedMemoryCarveout`
 * (which does not take into account `reservedSharedMemPerBlock`),
 * given by a desired schmem-L1 split and a per-block memory requirement in bytes.
 *
 * NB: As per the programming guide, the memory carveout setting is just a hint for the driver; it's
 * free to choose any shmem-L1 configuration it deems appropriate. For example, if you set the
 * carveout to zero, it will choose a non-zero config that will allow to run at least one active
 * block per SM.
 *
 * @param shmem_fraction
 *   a fraction representing a desired split (shmem / (shmem + L1)) [0, 1].
 * @param shmem_per_block
 *   a shared memory usage per block (dynamic + static shared memory sizes), in bytes.
 * @param dev_props
 *   device properties.
 * @return
 *   a carveout value in percents [0, 100].
 */
constexpr inline auto estimate_carveout(double shmem_fraction,
                                        size_t shmem_per_block,
                                        const cudaDeviceProp& dev_props) -> int
{
  using shmem_unit = raft::Pow2<128>;
  size_t m         = shmem_unit::roundUp(shmem_per_block);
  size_t r         = dev_props.reservedSharedMemPerBlock;
  size_t s         = dev_props.sharedMemPerMultiprocessor;
  return (size_t(100 * s * m * shmem_fraction) - (m - 1) * r) / (s * (m + r));
}

template <typename OutT>
auto get_out_type_tag()
{
  if constexpr (std::is_same_v<OutT, float>) {
    return tag_out_f{};
  } else if constexpr (std::is_same_v<OutT, half>) {
    return tag_out_h{};
  } else {
    static_assert(sizeof(OutT) == 0, "Unsupported OutT type");
  }
}

template <typename LutT>
auto get_lut_type_tag()
{
  if constexpr (std::is_same_v<LutT, float>) {
    return tag_lut_f{};
  } else if constexpr (std::is_same_v<LutT, half>) {
    return tag_lut_h{};
  } else if constexpr (std::is_same_v<LutT, cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, false>>) {
    return tag_lut_fp8_unsigned{};
  } else if constexpr (std::is_same_v<LutT, cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, true>>) {
    return tag_lut_fp8_signed{};
  } else {
    static_assert(sizeof(LutT) == 0, "Unsupported LutT type");
  }
}

template <typename FilterT>
auto get_filter_type_tag()
{
  using namespace cuvs::neighbors::filtering;

  // Determine the filter implementation tag
  if constexpr (std::is_same_v<FilterT, none_sample_filter>) { return tag_filter_none{}; }
  if constexpr (std::is_same_v<FilterT, bitset_filter<uint32_t, int64_t>>) {
    return tag_filter_bitset{};
  }
}

template <typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut,
          uint32_t PqBits,
          int Capacity,
          typename FilterT,
          typename MetricTag,
          bool IncrementScore>
auto kernel_try_capacity(uint32_t k_max)
{
// <<<<<<< OLD CODE from 815a9316d (f3458e8bb) - COMMENTED OUT
//   /* Shared memory:
//
//     * lut_scores: lookup table (LUT) of size = `pq_dim << PqBits`  (when EnableSMemLut)
//     * lut_end+:
//        * base_diff: size = dim (which is equal to `pq_dim * pq_len`)  or dim*2
//        * topk::warp_sort::mem_required - local topk temporary buffer (if necessary)
//     * topk::block_sort: some amount of shared memory, but overlaps with the rest:
//         block_sort only needs shared memory for `.done()` operation, which can come very last.
//   */
//   extern __shared__ __align__(256) uint8_t smem_buf[];  // NOLINT
//   constexpr bool kManageLocalTopK = Capacity > 0;
//
//   constexpr uint32_t PqShift = 1u << PqBits;  // NOLINT
//   constexpr uint32_t PqMask  = PqShift - 1u;  // NOLINT
//
//   const uint32_t pq_len   = dim / pq_dim;
//   const uint32_t lut_size = pq_dim * PqShift;
//
//   if constexpr (EnableSMemLut) {
//     lut_scores = reinterpret_cast<LutT*>(smem_buf);
//   } else {
//     lut_scores += lut_size * blockIdx.x;
//   }
//
//   uint8_t* lut_end = nullptr;
//   if constexpr (EnableSMemLut) {
//     lut_end = reinterpret_cast<uint8_t*>(lut_scores + lut_size);
//   } else {
//     lut_end = smem_buf;
//   }
//
//   for (int ib = blockIdx.x; ib < n_queries * n_probes; ib += gridDim.x) {
//     if (ib >= gridDim.x) {
//       // sync shared memory accesses on the second and further iterations
//       __syncthreads();
//     }
//     uint32_t query_ix;
//     uint32_t probe_ix;
//     if (index_list == nullptr) {
//       query_ix = ib % n_queries;
//       probe_ix = ib / n_queries;
//     } else {
//       auto ordered_ix = index_list[ib];
//       query_ix        = ordered_ix / n_probes;
//       probe_ix        = ordered_ix % n_probes;
//     }
//
//     const uint32_t* chunk_indices = _chunk_indices + (n_probes * query_ix);
//     const float* query            = queries + (dim * query_ix);
//     OutT* out_scores;
//     uint32_t* out_indices = nullptr;
//     if constexpr (kManageLocalTopK) {
//       // Store topk calculated distances to out_scores (and its indices to out_indices)
//       const uint64_t out_offset = probe_ix + n_probes * query_ix;
//       out_scores                = _out_scores + out_offset * topk;
//       out_indices               = _out_indices + out_offset * topk;
//     } else {
//       // Store all calculated distances to out_scores
//       out_scores = _out_scores + uint64_t(max_samples) * query_ix;
//     }
//     uint32_t label              = cluster_labels[n_probes * query_ix + probe_ix];
//     const float* cluster_center = cluster_centers + dim * label;
//     const float* pq_center;
//     if (codebook_kind == codebook_gen::PER_SUBSPACE) {
//       pq_center = pq_centers;
//     } else {
//       pq_center = pq_centers + (pq_len << PqBits) * label;
//     }
//
//     if constexpr (PrecompBaseDiff) {
//       // Reduce number of memory reads later by pre-computing parts of the score
//       switch (metric) {
//         case distance::DistanceType::L2SqrtExpanded:
//         case distance::DistanceType::L2Expanded: {
//           for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
//             reinterpret_cast<float*>(lut_end)[i] = query[i] - cluster_center[i];
//           }
//         } break;
//         case distance::DistanceType::CosineExpanded:
//         case distance::DistanceType::InnerProduct: {
//           float2 pvals;
//           for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
//             pvals.x                               = query[i];
//             pvals.y                               = cluster_center[i] * pvals.x;
//             reinterpret_cast<float2*>(lut_end)[i] = pvals;
//           }
//         } break;
//         default: __builtin_unreachable();
//       }
//       __syncthreads();
//     }
//
//     {
//       // Create a lookup table
//       // For each subspace, the lookup table stores the distance between the actual query vector
//       // (projected into the subspace) and all possible pq vectors in that subspace.
//       for (uint32_t i = threadIdx.x; i < lut_size; i += blockDim.x) {
//         const uint32_t i_pq  = i >> PqBits;
//         uint32_t j           = i_pq * pq_len;
//         const uint32_t j_end = pq_len + j;
//         auto cur_pq_center   = pq_center + (i & PqMask) +
//                              (codebook_kind == codebook_gen::PER_SUBSPACE ? j * PqShift : 0u);
//         float score = 0.0;
//         do {
//           float pq_c = *cur_pq_center;
//           cur_pq_center += PqShift;
//           switch (metric) {
//             case distance::DistanceType::L2SqrtExpanded:
//             case distance::DistanceType::L2Expanded: {
//               float diff;
//               if constexpr (PrecompBaseDiff) {
//                 diff = reinterpret_cast<float*>(lut_end)[j];
//               } else {
//                 diff = query[j] - cluster_center[j];
//               }
//               diff -= pq_c;
//               score += diff * diff;
//             } break;
//             case distance::DistanceType::CosineExpanded:
//             case distance::DistanceType::InnerProduct: {
//               // NB: we negate the scores as we hardcoded select-topk to always compute the minimum
//               float q;
//               if constexpr (PrecompBaseDiff) {
//                 float2 pvals = reinterpret_cast<float2*>(lut_end)[j];
//                 q            = pvals.x;
//                 score -= pvals.y;
//               } else {
//                 q = query[j];
//                 score -= q * cluster_center[j];
//               }
//               score -= q * pq_c;
//             } break;
//             default: __builtin_unreachable();
//           }
//         } while (++j < j_end);
//         lut_scores[i] = LutT(score);
//       }
//     }
//
//     // Define helper types for efficient access to the pq_dataset, which is stored in an interleaved
//     // format. The chunks of PQ data are stored in kIndexGroupVecLen-bytes-long chunks, interleaved
//     // in groups of kIndexGroupSize elems (which is normally equal to the warp size) for the fastest
//     // possible access by thread warps.
//     //
//     // Consider one record in the pq_dataset is `pq_dim * pq_bits`-bit-long.
//     // Assuming `kIndexGroupVecLen = 16`, one chunk of data read by a thread at once is 128-bits.
//     // Then, such a chunk contains `chunk_size = 128 / pq_bits` record elements, and the record
//     // consists of `ceildiv(pq_dim, chunk_size)` chunks. The chunks are interleaved in groups of 32,
//     // so that the warp can achieve the best coalesced read throughput.
//     using group_align  = raft::Pow2<kIndexGroupSize>;
//     using vec_align    = raft::Pow2<kIndexGroupVecLen>;
//     using local_topk_t = block_sort_t<Capacity, OutT, uint32_t>;
//     using op_t         = uint32_t;
//     using vec_t        = raft::TxN_t<op_t, kIndexGroupVecLen / sizeof(op_t)>;
//
//     uint32_t sample_offset = 0;
//     if (probe_ix > 0) { sample_offset = chunk_indices[probe_ix - 1]; }
//     uint32_t n_samples            = chunk_indices[probe_ix] - sample_offset;
//     uint32_t n_samples_aligned    = group_align::roundUp(n_samples);
//     constexpr uint32_t kChunkSize = (kIndexGroupVecLen * 8u) / PqBits;
//     uint32_t pq_line_width = raft::div_rounding_up_unsafe(pq_dim, kChunkSize) * kIndexGroupVecLen;
//
//     auto pq_thread_data = pq_dataset[label] + group_align::roundDown(threadIdx.x) * pq_line_width +
//                           group_align::mod(threadIdx.x) * vec_align::Value;
//     pq_line_width *= blockDim.x;
//
//     constexpr OutT kDummy = raft::upper_bound<OutT>();
//     OutT query_kth        = kDummy;
//     if constexpr (kManageLocalTopK) { query_kth = OutT(query_kths[query_ix]); }
//     OutT early_stop_limit = kDummy;
//     switch (metric) {
//       // If the metric is non-negative, we can use the query_kth approximation as an early stop
//       // threshold to skip some iterations when computing the score. Add such metrics here.
//       case distance::DistanceType::L2SqrtExpanded:
//       case distance::DistanceType::L2Expanded: {
//         early_stop_limit = query_kth;
//       } break;
//       default: break;
//     }
//
//     // Ensure lut_scores is written by all threads before using it in ivfpq-compute-score
//     __threadfence_block();
//     __syncthreads();
//     local_topk_t block_topk(topk, lut_end, query_kth);
//
//     // Compute a distance for each sample
//     for (uint32_t i = threadIdx.x; i < n_samples_aligned;
//          i += blockDim.x, pq_thread_data += pq_line_width) {
//       OutT score = kDummy;
//       bool valid = i < n_samples;
//       // Check bounds and that the sample is acceptable for the query
//       if (valid && sample_filter(queries_offset + query_ix, label, i)) {
//         score = ivfpq_compute_score<OutT, LutT, vec_t, PqBits>(
//           pq_dim,
//           reinterpret_cast<const vec_t::io_t*>(pq_thread_data),
//           lut_scores,
//           early_stop_limit);
//         if (metric == distance::DistanceType::CosineExpanded) { score = OutT(1) + score; }
//       }
//       if constexpr (kManageLocalTopK) {
//         block_topk.add(score, sample_offset + i);
//       } else {
//         if (valid) { out_scores[sample_offset + i] = score; }
//       }
// =======
  if constexpr (Capacity > 0) {
    if (k_max == 0 || k_max > Capacity) {
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 PqBits,
                                 0,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
// >>>>>>> END NEW CODE (f3458e8bb)
    }
  }
  if constexpr (Capacity > 1) {
    if (k_max * 2 <= Capacity) {
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 PqBits,
                                 (Capacity / 2),
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    }
  }

  using out_tag    = decltype(get_out_type_tag<OutT>());
  using lut_tag    = decltype(get_lut_type_tag<LutT>());
  using filter_tag = decltype(get_filter_type_tag<FilterT>());
  using precomp_base_diff_metric_tag =
    std::conditional_t<PrecompBaseDiff, MetricTag, tag_metric_none>;

  constexpr bool kManageLocalTopK = Capacity > 0;

  ComputeSimilarityPlanner planner;
  planner.add_entrypoint<out_tag, lut_tag>();
  planner.add_prepare_lut_function<lut_tag, EnableSMemLut, PqBits>();
  planner.add_store_calculated_distances_function<out_tag, kManageLocalTopK>();
  planner.add_precompute_base_diff_function<precomp_base_diff_metric_tag>();
  planner.add_create_lut_function<lut_tag, MetricTag, PrecompBaseDiff, PqBits>();
  planner.add_compute_distances_function<out_tag, lut_tag, Capacity>();
  planner.add_get_early_stop_limit_function<out_tag, MetricTag>();
  planner.add_sample_filter_function<filter_tag>();
  planner.add_get_line_width_function<PqBits>();
  planner.add_compute_score_function<out_tag, lut_tag, PqBits>();
  planner.add_increment_score_function<out_tag, IncrementScore>();
  return planner.get_launcher();
}

template <typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut,
          typename FilterT,
          typename MetricTag,
          bool IncrementScore>
auto get_compute_similarity_launcher(uint32_t pq_bits, uint32_t k_max)
{
  switch (pq_bits) {
    case 4:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 4,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 5:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 5,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 6:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 6,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 7:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 7,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 8:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 8,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
  }
}

/** Estimate the occupancy for the given kernel on the given device. */
struct occupancy_t {
  using shmem_unit = raft::Pow2<128>;

  int blocks_per_sm = 0;
  double occupancy  = 0.0;
  double shmem_use  = 1.0;

  inline occupancy_t() = default;
  inline occupancy_t(size_t smem,
                     uint32_t n_threads,
                     cudaKernel_t kernel,
                     const cudaDeviceProp& dev_props)
  {
    RAFT_CUDA_TRY(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, n_threads, smem));
    occupancy = double(blocks_per_sm * n_threads) / double(dev_props.maxThreadsPerMultiProcessor);
    shmem_use = double(shmem_unit::roundUp(smem) * blocks_per_sm) /
                double(dev_props.sharedMemPerMultiprocessor);
  }
};

template <typename OutT, typename LutT>
void compute_similarity_run(selected<OutT, LutT> s,
                            rmm::cuda_stream_view stream,
                            uint32_t dim,
                            uint32_t n_probes,
                            uint32_t pq_dim,
                            uint32_t n_queries,
                            uint32_t queries_offset,
                            codebook_gen codebook_kind,
                            uint32_t topk,
                            uint32_t max_samples,
                            const float* cluster_centers,
                            const float* pq_centers,
                            const uint8_t* const* pq_dataset,
                            const uint32_t* cluster_labels,
                            const uint32_t* _chunk_indices,
                            const float* queries,
                            const uint32_t* index_list,
                            float* query_kths,
                            const int64_t* const* inds_ptrs,
                            uint32_t* bitset_ptr,
                            int64_t bitset_len,
                            int64_t original_nbits,
                            LutT* lut_scores,
                            OutT* _out_scores,
                            uint32_t* _out_indices)
{
// <<<<<<< OLD CODE from 815a9316d (f3458e8bb) - COMMENTED OUT
//   auto launch_kernel = [&](filtering::ivf_filter_dev sample_filter) {
//     auto kernel =
//       reinterpret_cast<compute_similarity_kernel_t<OutT, LutT>>(const_cast<void*>(get_kernel(s)));
//     kernel<<<s.grid_dim, s.block_dim, s.smem_size, stream>>>(dim,
//                                                              n_probes,
//                                                              pq_dim,
//                                                              n_queries,
//                                                              queries_offset,
//                                                              metric,
//                                                              codebook_kind,
//                                                              topk,
//                                                              max_samples,
//                                                              cluster_centers,
//                                                              pq_centers,
//                                                              pq_dataset,
//                                                              cluster_labels,
//                                                              _chunk_indices,
//                                                              queries,
//                                                              index_list,
//                                                              query_kths,
//                                                              sample_filter,
//                                                              lut_scores,
//                                                              _out_scores,
//                                                              _out_indices);
//     RAFT_CHECK_CUDA(stream);
//   };
//
//   switch (sample_filter_ref.get_filter_type()) {
//     case filtering::FilterType::None: {
//       try {
//         auto& typed_sample_filter = dynamic_cast<
//           const filtering::ivf_to_sample_filter<int64_t, filtering::none_sample_filter>&>(
//           sample_filter_ref);
//         filtering::ivf_filter_dev sample_filter{filtering::none_filter_args_t{}};
//         launch_kernel(sample_filter);
//       } catch (const std::bad_cast& e) {
//       }
//       break;
//     }
//     case filtering::FilterType::Bitset: {
//       try {
//         auto& typed_sample_filter = dynamic_cast<
//           const filtering::ivf_to_sample_filter<int64_t,
//                                                 filtering::bitset_filter<uint32_t, int64_t>>&>(
//           sample_filter_ref);
//         filtering::ivf_filter_dev sample_filter{filtering::bitset_filter_args_t{
//           typed_sample_filter.inds_ptrs_, typed_sample_filter.next_filter_.bitset_view_}};
//         launch_kernel(sample_filter);
//       } catch (const std::bad_cast& e) {
//       }
//       break;
//     }
//     default: RAFT_FAIL("Unsupported filter type");
//   }
// =======
  s.launcher->template dispatch<compute_similarity_func_t<OutT, LutT>>(stream,
                                                                       s.grid_dim,
                                                                       s.block_dim,
                                                                       s.smem_size,
                                                                       dim,
                                                                       n_probes,
                                                                       pq_dim,
                                                                       n_queries,
                                                                       queries_offset,
                                                                       codebook_kind,
                                                                       topk,
                                                                       max_samples,
                                                                       cluster_centers,
                                                                       pq_centers,
                                                                       pq_dataset,
                                                                       cluster_labels,
                                                                       _chunk_indices,
                                                                       queries,
                                                                       index_list,
                                                                       query_kths,
                                                                       inds_ptrs,
                                                                       bitset_ptr,
                                                                       bitset_len,
                                                                       original_nbits,
                                                                       lut_scores,
                                                                       _out_scores,
                                                                       _out_indices);
  RAFT_CHECK_CUDA(stream);
// >>>>>>> END NEW CODE (f3458e8bb)
}

/**
 * Use heuristics to choose an optimal instance of the search kernel.
 * It selects among a few kernel variants (with/out using shared mem for
 * lookup tables / precomputed distances) and tries to choose the block size
 * to maximize kernel occupancy.
 *
 * @param manage_local_topk
 *    whether use the fused calculate+select or just calculate the distances for each
 *    query and probed cluster.
 *
 * @param locality_hint
 *    beyond this limit do not consider increasing the number of active blocks per SM
 *    would improve locality anymore.
 */
template <typename OutT, typename LutT, typename FilterT, typename MetricTag, bool IncrementScore>
auto compute_similarity_select(const cudaDeviceProp& dev_props,
                               bool manage_local_topk,
                               int locality_hint,
                               double preferred_shmem_carveout,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               uint32_t precomp_data_count,
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk) -> selected<OutT, LutT>
{
  // Shared memory for storing the lookup table
  size_t lut_mem = sizeof(LutT) * (pq_dim << pq_bits);
  // Shared memory for storing pre-computed pieces to speedup the lookup table construction
  // (e.g. the distance between a cluster center and the query for L2).
  size_t bdf_mem = sizeof(float) * precomp_data_count;

  // Shared memory used by the fused top-k during cluster scanning;
  // may overlap with the precomputed distance array
  struct ltk_add_mem_t {
    size_t (*mem_required)(uint32_t);

    ltk_add_mem_t(bool manage_local_topk, uint32_t topk)
      : mem_required(pq_block_sort<kMaxCapacity, OutT, uint32_t>::get_mem_required(
          manage_local_topk ? topk : 0))
    {
    }

    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return mem_required(n_threads);
    }
  } ltk_add_mem{manage_local_topk, topk};

  // Shared memory for the fused top-k component;
  // may overlap with all other uses of shared memory
  struct ltk_reduce_mem_t {
    uint32_t subwarp_size;
    uint32_t topk;
    bool manage_local_topk;
    ltk_reduce_mem_t(bool manage_local_topk, uint32_t topk, int warp_size)
      : manage_local_topk(manage_local_topk), topk(topk)
    {
      subwarp_size = warp_size;
      while (topk * 2 <= subwarp_size) {
        subwarp_size /= 2;
      }
    }

    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return manage_local_topk ? raft::matrix::detail::select::warpsort::
                                   template calc_smem_size_for_block_wide<OutT, uint32_t>(
                                     n_threads / subwarp_size, topk)
                               : 0;
    }
  } ltk_reduce_mem{manage_local_topk, topk, dev_props.warpSize};

  struct total_shared_mem_t {
    ltk_add_mem_t& ltk_add_mem;
    ltk_reduce_mem_t& ltk_reduce_mem;
    size_t lut_mem;
    size_t bdf_mem;
    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return std::max(ltk_reduce_mem(n_threads),
                      lut_mem + std::max(bdf_mem, ltk_add_mem(n_threads)));
    }
  };

  // Total amount of work; should be enough to occupy the GPU.
  uint32_t n_blocks = n_queries * n_probes;

  // The minimum block size we may want:
  //   1. It's a power-of-two for efficient L1 caching of pq_centers values
  //      (multiples of `1 << pq_bits`).
  //   2. It should be large enough to fully utilize an SM.
  //   3. It must be at least kIndexGroupSize for correct interleaved data access.
  uint32_t n_threads_min = kIndexGroupSize;
  while (dev_props.maxBlocksPerMultiProcessor * int(n_threads_min) <
         dev_props.maxThreadsPerMultiProcessor) {
    n_threads_min *= 2;
  }
  // Further increase the minimum block size to make sure full device occupancy
  // (NB: this may lead to `n_threads_min` being larger than the kernel's maximum)
  while (int(n_blocks * n_threads_min) <
           dev_props.multiProcessorCount * dev_props.maxThreadsPerMultiProcessor &&
         int(n_threads_min) < dev_props.maxThreadsPerBlock) {
    n_threads_min *= 2;
  }
  // Even further, increase it to allow less blocks per SM if there not enough queries.
  // With this, we reduce the chance of different clusters being processed by two blocks
  // on the same SM and thus improve the data locality for L1 caching.
  while (int(n_queries * n_threads_min) < dev_props.maxThreadsPerMultiProcessor &&
         int(n_threads_min) < dev_props.maxThreadsPerBlock) {
    n_threads_min *= 2;
  }

  // Granularity of changing the number of threads when computing the maximum block size.
  // It's good to have it multiple of the PQ book width.
  uint32_t n_threads_gty = raft::round_up_safe<uint32_t>(1u << pq_bits, dev_props.warpSize);

  /*
   Shared memory / L1 cache balance is the main limiter of this kernel.
   The more blocks per SM we launch, the more shared memory we need. Besides that, we have
   three versions of the kernel varying in performance and shmem usage.

   We try the most demanding and the fastest kernel first, trying to maximize occupancy with
   the minimum number of blocks (just one, really). Then, we tweak the `n_threads` to further
   optimize occupancy and data locality for the L1 cache.
   */
  auto topk_or_zero = manage_local_topk ? topk : 0u;
  auto conf_fast =
    get_compute_similarity_launcher<OutT, LutT, true, true, FilterT, MetricTag, IncrementScore>(
      pq_bits, topk_or_zero);
  auto conf_no_basediff =
    get_compute_similarity_launcher<OutT, LutT, false, true, FilterT, MetricTag, IncrementScore>(
      pq_bits, topk_or_zero);
  auto conf_no_smem_lut =
    get_compute_similarity_launcher<OutT, LutT, true, false, FilterT, MetricTag, IncrementScore>(
      pq_bits, topk_or_zero);
  std::array candidates{
    std::make_tuple(
      conf_fast, total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, lut_mem, bdf_mem}, true),
    std::make_tuple(
      conf_no_basediff, total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, lut_mem, 0}, true),
    std::make_tuple(
      conf_no_smem_lut, total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, 0, bdf_mem}, false)};

  // we may allow slightly lower than 100% occupancy;
  constexpr double kTargetOccupancy = 0.75;
  // This struct is used to select the better candidate
  occupancy_t selected_perf{};
  selected<OutT, LutT> selected_config;
// <<<<<<< OLD CODE from 815a9316d (f3458e8bb) - COMMENTED OUT
//   for (auto [kernel, smem_size_f, lut_is_in_shmem] : candidates) {
//     if (smem_size_f(dev_props.warpSize) > dev_props.sharedMemPerBlockOptin) {
// =======
  for (auto [launcher, smem_size_f, lut_is_in_shmem] : candidates) {
    if (smem_size_f(raft::WarpSize) > dev_props.sharedMemPerBlockOptin) {
// >>>>>>> END NEW CODE (f3458e8bb)
      // Even a single block cannot fit into an SM due to shmem requirements. Skip the candidate.
      continue;
    }

    auto kernel = launcher->get_kernel();

    // First, we set the carveout hint to the preferred value. The driver will increase this if
    // needed to run at least one block per SM. At the same time, if more blocks fit into one SM,
    // this carveout value will limit the calculated occupancy. When we're done selecting the best
    // launch configuration, we will tighten the carveout once more, based on the final memory
    // usage and occupancy.
    const int max_carveout =
      estimate_carveout(preferred_shmem_carveout, smem_size_f(dev_props.warpSize), dev_props);
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, max_carveout));

    // Get the theoretical maximum possible number of threads per block
    cudaFuncAttributes kernel_attrs;
    RAFT_CUDA_TRY(cudaFuncGetAttributes(&kernel_attrs, reinterpret_cast<const void*>(kernel)));
    uint32_t n_threads =
      raft::round_down_safe<uint32_t>(kernel_attrs.maxThreadsPerBlock, n_threads_gty);

    // Actual required shmem depens on the number of threads
    size_t smem_size = smem_size_f(n_threads);

    // Make sure the kernel can get enough shmem.
    cudaError_t cuda_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cuda_status != cudaSuccess) {
      RAFT_EXPECTS(
        cuda_status == cudaGetLastError(),
        "Tried to reset the expected cuda error code, but it didn't match the expectation");
      // Failed to request enough shmem for the kernel. Skip the candidate.
      continue;
    }

    occupancy_t cur(smem_size, n_threads, kernel, dev_props);
    if (cur.blocks_per_sm <= 0) {
      // For some reason, we still cannot make this kernel run. Skip the candidate.
      continue;
    }

    {
      // Try to reduce the number of threads to increase occupancy and data locality
      auto n_threads_tmp = n_threads_min;
      while (n_threads_tmp * 2 < n_threads) {
        n_threads_tmp *= 2;
      }
      if (n_threads_tmp < n_threads) {
        while (n_threads_tmp >= n_threads_min) {
          auto smem_size_tmp = smem_size_f(n_threads_tmp);
          occupancy_t tmp(smem_size_tmp, n_threads_tmp, kernel, dev_props);
          bool select_it = false;
          if (lut_is_in_shmem && locality_hint >= tmp.blocks_per_sm) {
            // Normally, the smaller the block the better for L1 cache hit rate.
            // Hence, the occupancy should be "just good enough"
            select_it = tmp.occupancy >= min(kTargetOccupancy, cur.occupancy);
          } else if (lut_is_in_shmem) {
            // If we don't have enough repeating probes (locality_hint < tmp.blocks_per_sm),
            // the locality is not going to improve with increasing the number of blocks per SM.
            // Hence, the only metric here is the occupancy.
            bool improves_occupancy = tmp.occupancy > cur.occupancy;
            // Otherwise, the performance still improves with a smaller block size,
            // given there is enough work to do
            bool improves_parallelism =
              tmp.occupancy == cur.occupancy &&
              7u * tmp.blocks_per_sm * dev_props.multiProcessorCount <= n_blocks;
            select_it = improves_occupancy || improves_parallelism;
          } else {
            // If we don't use shared memory for the lookup table, increasing the number of blocks
            // is very taxing on the global memory usage.
            // In this case, the occupancy must increase a lot to make it worth the cost.
            select_it = tmp.occupancy >= min(1.0, cur.occupancy / kTargetOccupancy);
          }
          if (select_it) {
            n_threads = n_threads_tmp;
            smem_size = smem_size_tmp;
            cur       = tmp;
          }
          n_threads_tmp /= 2;
        }
      }
    }

    {
      if (selected_perf.occupancy <= 0.0  // no candidate yet
          || (selected_perf.occupancy < cur.occupancy * kTargetOccupancy &&
              selected_perf.shmem_use >= cur.shmem_use)  // much improved occupancy
      ) {
        selected_perf = cur;
        if (lut_is_in_shmem) {
          selected_config = selected<OutT, LutT>(
            std::move(launcher), dim3(n_blocks, 1, 1), dim3(n_threads, 1, 1), smem_size, size_t(0));
        } else {
          // When the global memory is used for the lookup table, we need to minimize the grid
          // size; otherwise, the kernel may quickly run out of memory.
          auto n_blocks_min =
            std::min<uint32_t>(n_blocks, cur.blocks_per_sm * dev_props.multiProcessorCount);
          selected_config = selected<OutT, LutT>(std::move(launcher),
                                                 dim3(n_blocks_min, 1, 1),
                                                 dim3(n_threads, 1, 1),
                                                 smem_size,
                                                 size_t(n_blocks_min) * size_t(pq_dim << pq_bits));
        }
        // Actual shmem/L1 split wildly rounds up the specified preferred carveout, so we set here
        // a rather conservative bar; most likely, the kernel gets more shared memory than this,
        // and the occupancy doesn't get hurt.
        auto carveout = std::min<int>(max_carveout, std::ceil(100.0 * cur.shmem_use));
        RAFT_CUDA_TRY(
          cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
        if (cur.occupancy >= kTargetOccupancy) { break; }
      } else if (selected_perf.occupancy > 0.0) {
        // If we found a reasonable candidate on a previous iteration, and this one is not better,
        // then don't try any more candidates because they are much slower anyway.
        break;
      }
    }
  }

  RAFT_EXPECTS(selected_perf.occupancy > 0.0,
               "Couldn't determine a working kernel launch configuration.");

  return selected_config;
}

}  // namespace cuvs::neighbors::ivf_pq::detail
