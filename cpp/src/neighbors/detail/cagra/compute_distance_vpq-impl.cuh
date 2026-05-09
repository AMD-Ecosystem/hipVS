/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "compute_distance_vpq.hpp"
#include "device_common.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
struct cagra_q_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type   = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using CODE_BOOK_T = CodebookT;
  using QUERY_T     = QueryT;
  using base_type::args;
  using base_type::extra_ptr3;
  using typename base_type::args_t;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;
  constexpr static inline auto kPqBits          = PQ_BITS;
  constexpr static inline auto kPqLen           = PQ_LEN;

  static_assert(std::is_same_v<CODE_BOOK_T, half>, "Only CODE_BOOK_T = `half` is supported now");

  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_ptr(args_t& args) noexcept
    -> const uint8_t*&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto vq_code_book_ptr(args_t& args) noexcept
    -> const CODE_BOOK_T*&
  {
    return (const CODE_BOOK_T*&)args.extra_ptr2;
  }
  RAFT_INLINE_FUNCTION constexpr auto pq_code_book_ptr() noexcept -> const CODE_BOOK_T*&
  {
    return (const CODE_BOOK_T*&)extra_ptr3;
  }
  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_dim(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }

  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_ptr(const args_t& args) noexcept
    -> const uint8_t* const&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto vq_code_book_ptr(const args_t& args) noexcept
    -> const CODE_BOOK_T* const&
  {
    return (const CODE_BOOK_T*&)args.extra_ptr2;
  }
  RAFT_INLINE_FUNCTION constexpr auto pq_code_book_ptr() const noexcept -> const CODE_BOOK_T* const&
  {
    return (const CODE_BOOK_T*&)extra_ptr3;
  }
  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_dim(const args_t& args) noexcept
    -> const uint32_t&
  {
    return args.extra_word1;
  }

  static constexpr std::uint32_t kSMemCodeBookSizeInBytes =
    (1 << PQ_BITS) * PQ_LEN * utils::size_of<CODE_BOOK_T>();

  _RAFT_HOST_DEVICE cagra_q_dataset_descriptor_t(const std::uint8_t* encoded_dataset_ptr,
                                                 std::uint32_t encoded_dataset_dim,
                                                 const CODE_BOOK_T* vq_code_book_ptr,
                                                 const CODE_BOOK_T* pq_code_book_ptr,
                                                 IndexT size,
                                                 std::uint32_t dim)
    : base_type(size, dim, raft::Pow2<TeamSize>::Log2, get_smem_ws_size_in_bytes(dim))
  {
    cagra_q_dataset_descriptor_t::encoded_dataset_ptr(args) = encoded_dataset_ptr;
    cagra_q_dataset_descriptor_t::vq_code_book_ptr(args)    = vq_code_book_ptr;
    this->pq_code_book_ptr()                                = pq_code_book_ptr;
    cagra_q_dataset_descriptor_t::encoded_dataset_dim(args) = encoded_dataset_dim;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(cagra_q_dataset_descriptor_t) == alignof(base_type));
  }

  // Public static method to compute smem size without constructing descriptor
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    /* SMEM workspace layout:
      1. The descriptor itself
      2. Codebook (kSMemCodeBookSizeInBytes bytes)
      3. Queries (smem_query_buffer_length elems)
    */
    return sizeof(cagra_q_dataset_descriptor_t) + kSMemCodeBookSizeInBytes +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }
// <<<<<<< OLD CODE from 60b6ef05c (86bc514df) - COMMENTED OUT
// };
//
// template <auto Block, auto Stride, typename T>
// RAFT_DEVICE_INLINE_FUNCTION constexpr auto transpose(T x) -> T
// {
//   auto i = x % Block;
//   auto j = x / Block;
//   auto k = i % Stride;
//   auto l = i / Stride;
//   return j * Block + k * (Block / Stride) + l;
// }
//
// template <typename DescriptorT>
// _RAFT_DEVICE __noinline__ auto setup_workspace_vpq(const DescriptorT* that,
//                                                    void* smem_ptr,
//                                                    const typename DescriptorT::DATA_T* queries_ptr,
//                                                    uint32_t query_id) -> const DescriptorT*
// {
//   using QUERY_T                   = typename DescriptorT::QUERY_T;
//   using CODE_BOOK_T               = typename DescriptorT::CODE_BOOK_T;
//   using word_type                 = uint32_t;
//   constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
//   constexpr auto PQ_BITS          = DescriptorT::kPqBits;
//   constexpr auto PQ_LEN           = DescriptorT::kPqLen;
//
//   auto* r = reinterpret_cast<DescriptorT*>(smem_ptr);
//
//   if (r != that) {
//     constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
//     using blob_type           = word_type[kCount];
//     auto& src                 = reinterpret_cast<const blob_type&>(*that);
//     auto& dst                 = reinterpret_cast<blob_type&>(*r);
//     for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
//       dst[i] = src[i];
//     }
// #ifdef __HIP_PLATFORM_AMD__
//     auto codebook_buf = device::flat_to_shared_offset(r + 1);
// #else
//     auto codebook_buf = uint32_t(__cvta_generic_to_shared(r + 1));
// #endif
//     const auto smem_ptr_offset =
//       reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
//     if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
//       r->args.smem_ws_ptr = codebook_buf;
//     }
//     __syncthreads();
//
//     // Copy PQ table
//     for (unsigned i = threadIdx.x * 2; i < (1 << PQ_BITS) * PQ_LEN; i += blockDim.x * 2) {
//       half2 buf2;
//       buf2.x = r->pq_code_book_ptr()[i];
//       buf2.y = r->pq_code_book_ptr()[i + 1];
//
//       // Change the order of PQ code book array to reduce the
//       // frequency of bank conflicts.
//       constexpr auto num_elements_per_bank  = 4 / utils::size_of<CODE_BOOK_T>();
//       constexpr auto num_banks_per_subspace = PQ_LEN / num_elements_per_bank;
//       const auto j                          = i / num_elements_per_bank;
//       const auto smem_index =
//         (j / num_banks_per_subspace) + (j % num_banks_per_subspace) * (1 << PQ_BITS);
//
//       device::sts(codebook_buf + smem_index * sizeof(half2), buf2);
//     }
//   }
// =======
// >>>>>>> END NEW CODE (86bc514df)

 private:
};

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
RAFT_KERNEL __launch_bounds__(1, 1)
  vpq_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                     const std::uint8_t* encoded_dataset_ptr,
                                     uint32_t encoded_dataset_dim,
                                     const CodebookT* vq_code_book_ptr,
                                     const CodebookT* pq_code_book_ptr,
                                     IndexT size,
                                     uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodebookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT,
                                                 half>;
  new (out) desc_type(
    encoded_dataset_ptr, encoded_dataset_dim, vq_code_book_ptr, pq_code_book_ptr, size, dim);
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
dataset_descriptor_host<DataT, IndexT, DistanceT>
vpq_descriptor_spec<Metric,
                    TeamSize,
                    DatasetBlockDim,
                    PqBits,
                    PqLen,
                    CodebookT,
                    DataT,
                    IndexT,
                    DistanceT>::init_(const cagra::search_params& params,
                                      const std::uint8_t* encoded_dataset_ptr,
                                      uint32_t encoded_dataset_dim,
                                      const CodebookT* vq_code_book_ptr,
                                      const CodebookT* pq_code_book_ptr,
                                      IndexT size,
                                      uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodebookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT,
                                                 half>;

  return host_type{
    desc_type{
      encoded_dataset_ptr, encoded_dataset_dim, vq_code_book_ptr, pq_code_book_ptr, size, dim},
    [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
        rmm::cuda_stream_view stream) {
      vpq_dataset_descriptor_init_kernel<Metric,
                                         TeamSize,
                                         DatasetBlockDim,
                                         PqBits,
                                         PqLen,
                                         CodebookT,
                                         DataT,
                                         IndexT,
                                         DistanceT><<<1, 1, 0, stream>>>(dev_ptr,
                                                                         encoded_dataset_ptr,
                                                                         encoded_dataset_dim,
                                                                         vq_code_book_ptr,
                                                                         pq_code_book_ptr,
                                                                         size,
                                                                         dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    },
    Metric,
    DatasetBlockDim,
    true,    // is_vpq
    PqBits,  // pq_bits
    PqLen};  // pq_len
}

}  // namespace cuvs::neighbors::cagra::detail
