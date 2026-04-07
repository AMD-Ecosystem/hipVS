// MIT License
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Fused Distance NN epilogue for CK Tile GEMM: computes distances from norms,
// performs row-wise reduction to find minimum (value, index) pairs, and
// atomically updates the global output buffer.
//
// This extends PairwiseDistanceCkEpilogue by:
// 1. Computing distances from GEMM output + norms (same as pairwise)
// 2. Row-wise reduction within the tile to find min(distance, column_index)
// 3. Atomic update to global min buffer using mutex pattern
//
// Key components:
// - Intra-thread reduction: each thread finds local min across its columns
// - Intra-warp reduction: warp shuffle to reduce across threads in same row
// - Cross-warp reduction: LDS for collecting per-warp row minima
// - Atomic update: mutex-protected update to global output

#pragma once

#include <algorithm>
#include <ck_tile/core.hpp>
#include <ck_tile/core/tensor/tile_distribution_encoding.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <ck_tile/ops/common/utils.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>
#include <limits>
#include <raft/core/kvp.hpp>

namespace cuvs::distance::detail {

// Problem config matching the pairwise epilogue structure
template <typename AccDataType_,
          typename ODataType_,
          typename IdxType_,
          ck_tile::index_t kM_,
          ck_tile::index_t kN_,
          ck_tile::index_t MWave_,
          ck_tile::index_t NWave_,
          ck_tile::index_t MPerXdl_,
          ck_tile::index_t NPerXdl_,
          ck_tile::index_t KPerXdl_,
          bool FixedVectorSize_                 = false,
          ck_tile::index_t VectorSizeC_         = 1,
          ck_tile::index_t BlockedXDLN_PerWarp_ = 1>
struct FusedDistanceNNCkEpilogueProblem {
  using AccDataType = ck_tile::remove_cvref_t<AccDataType_>;
  using ODataType   = ck_tile::remove_cvref_t<ODataType_>;
  using IdxType     = ck_tile::remove_cvref_t<IdxType_>;

  static constexpr ck_tile::index_t kBlockSize  = MWave_ * NWave_ * ck_tile::get_warp_size();
  static constexpr ck_tile::index_t kMPerBlock  = kM_;
  static constexpr ck_tile::index_t kNPerBlock  = kN_;
  static constexpr ck_tile::index_t MWave       = MWave_;
  static constexpr ck_tile::index_t NWave       = NWave_;
  static constexpr ck_tile::index_t MPerXdl     = MPerXdl_;
  static constexpr ck_tile::index_t NPerXdl     = NPerXdl_;
  static constexpr ck_tile::index_t KPerXdl     = KPerXdl_;
  static constexpr bool FixedVectorSize         = FixedVectorSize_;
  static constexpr ck_tile::index_t VectorSizeC = VectorSizeC_;
  static constexpr ck_tile::index_t BlockedXDLN_PerWarp = BlockedXDLN_PerWarp_;

  static constexpr ck_tile::index_t MPerIteration = MPerXdl * MWave;
  static constexpr ck_tile::index_t NPerIteration = NPerXdl * NWave;
  static constexpr ck_tile::index_t MRepeat       = kMPerBlock / (MPerXdl * MWave);
  static constexpr ck_tile::index_t NRepeat       = kNPerBlock / (NPerXdl * NWave);

  using ELayout = ck_tile::tensor_layout::gemm::RowMajor;
};

// Fused Distance NN Epilogue: distances + row-wise reduction + atomic update
template <typename Problem_, typename ElementWiseOp_, typename ReduceOpT_, typename KVPReduceOpT_>
struct FusedDistanceNNCkEpilogue {
  using Problem       = ck_tile::remove_cvref_t<Problem_>;
  using ElementWiseOp = ck_tile::remove_cvref_t<ElementWiseOp_>;
  using ReduceOpT     = ck_tile::remove_cvref_t<ReduceOpT_>;
  using KVPReduceOpT  = ck_tile::remove_cvref_t<KVPReduceOpT_>;
  using AccDataType   = ck_tile::remove_cvref_t<typename Problem::AccDataType>;
  using ODataType     = ck_tile::remove_cvref_t<typename Problem::ODataType>;
  using IdxType       = ck_tile::remove_cvref_t<typename Problem::IdxType>;
  using KVPair        = raft::KeyValuePair<IdxType, ODataType>;

  static constexpr ck_tile::index_t kBlockSize          = Problem::kBlockSize;
  static constexpr ck_tile::index_t kMPerBlock          = Problem::kMPerBlock;
  static constexpr ck_tile::index_t kNPerBlock          = Problem::kNPerBlock;
  static constexpr ck_tile::index_t MWave               = Problem::MWave;
  static constexpr ck_tile::index_t NWave               = Problem::NWave;
  static constexpr ck_tile::index_t MPerXdl             = Problem::MPerXdl;
  static constexpr ck_tile::index_t NPerXdl             = Problem::NPerXdl;
  static constexpr ck_tile::index_t MPerIteration       = Problem::MPerIteration;
  static constexpr ck_tile::index_t NPerIteration       = Problem::NPerIteration;
  static constexpr ck_tile::index_t MRepeat             = Problem::MRepeat;
  static constexpr ck_tile::index_t NRepeat             = Problem::NRepeat;
  static constexpr ck_tile::index_t BlockedXDLN_PerWarp = Problem::BlockedXDLN_PerWarp;

  using ATypeToUse = ODataType;
  using BTypeToUse = ODataType;
  using WG         = ck_tile::WarpGemmDispatcher<ATypeToUse,
                                                 BTypeToUse,
                                                 AccDataType,
                                                 MPerXdl,
                                                 NPerXdl,
                                                 Problem::KPerXdl,
                                                 false>;

  using CWarpDstr   = typename WG::CWarpDstr;
  using CWarpTensor = typename WG::CWarpTensor;

  static constexpr ck_tile::index_t GetVectorSizeC()
  {
    if constexpr (Problem::FixedVectorSize) return Problem::VectorSizeC;
    constexpr ck_tile::index_t max_vector_size = 16;
    return std::min(static_cast<int>(NPerIteration),
                    static_cast<int>(max_vector_size / sizeof(ODataType)));
  }

  static constexpr auto shuffle_tile_tuple = [] {
    constexpr ck_tile::index_t elem_per_thread = MPerXdl * NPerXdl / ck_tile::get_warp_size();
    if constexpr (elem_per_thread >= GetVectorSizeC()) {
      return std::make_tuple(1, 1);
    } else {
      constexpr ck_tile::index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
      return std::make_tuple(std::min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
    }
  }();
  static constexpr ck_tile::index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
  static constexpr ck_tile::index_t NumNXdlPerWavePerShuffle =
    std::max(BlockedXDLN_PerWarp, std::get<1>(shuffle_tile_tuple));

  static constexpr auto MNPerIterationShuffle = [] {
    constexpr ck_tile::index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;
    constexpr ck_tile::index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;
    if constexpr (kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
      return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);
    else
      return std::make_tuple(m_val, n_val);
  }();
  static constexpr ck_tile::index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);
  static constexpr ck_tile::index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);

  using SFC =
    ck_tile::space_filling_curve<ck_tile::sequence<kMPerBlock, kNPerBlock>,
                                 ck_tile::sequence<0, 1>,
                                 ck_tile::sequence<MPerIterationShuffle, NPerIterationShuffle>>;

  // LDS size for distance tile shuffle + reduction workspace
  // We need space for:
  // 1. Distance tile shuffle: MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType)
  // 2. Cross-warp reduction: kMPerBlock * NWave * sizeof(KVPair)
  __host__ __device__ static constexpr ck_tile::index_t GetSmemSize()
  {
    constexpr ck_tile::index_t shuffle_size =
      MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
    constexpr ck_tile::index_t reduce_size = kMPerBlock * NWave * sizeof(KVPair);
    return ck_tile::max(shuffle_size, reduce_size);
  }

  ElementWiseOp elfunc_;
  mutable ReduceOpT red_op_;
  mutable KVPReduceOpT pair_red_op_;

  __device__ FusedDistanceNNCkEpilogue(ElementWiseOp elfunc     = {},
                                       ReduceOpT red_op         = {},
                                       KVPReduceOpT pair_red_op = {})
    : elfunc_(elfunc), red_op_(red_op), pair_red_op_(pair_red_op)
  {
  }

  // Main operator: compute distances, reduce per row, atomic update
  template <typename OutT, typename OAccTile>
  __device__ void operator()(OutT* min_output,
                             int* mutex,
                             OAccTile const& o_acc_tile,
                             void* p_smem,
                             ODataType const* norm_x_ptr,
                             ODataType const* norm_y_ptr,
                             ck_tile::index_t i_m,
                             ck_tile::index_t i_n,
                             ck_tile::index_t M,
                             ck_tile::index_t N,
                             ODataType maxVal) const
  {
    using TileEncodingPattern = ck_tile::tile_distribution_encoding_pattern_2d<
      kBlockSize,
      MPerIterationShuffle,
      NPerIterationShuffle,
      GetVectorSizeC(),
      ck_tile::tile_distribution_pattern::thread_raked,
      1>;
    constexpr auto LdsTileDistr =
      ck_tile::make_static_tile_distribution(MakeLdsDistributionEncode());
    constexpr auto dram_tile_distribution = TileEncodingPattern::make_2d_static_tile_distribution();

    constexpr auto lds_block_desc = MakeLdsBlockDescriptor();
    auto o_lds_block              = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
      static_cast<ODataType*>(p_smem), lds_block_desc);

    auto lds_tile = ck_tile::make_static_distributed_tensor<AccDataType>(LdsTileDistr);

    auto in_lds_window =
      ck_tile::make_tile_window(o_lds_block,
                                ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                                                    ck_tile::number<NPerIterationShuffle>{}),
                                {0, 0},
                                LdsTileDistr);

    auto out_lds_window =
      ck_tile::make_tile_window(o_lds_block,
                                ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                                                    ck_tile::number<NPerIterationShuffle>{}),
                                {0, 0});

    // 1D norm vectors as (M,N) views with broadcast strides
    auto const norm_x_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
      norm_x_ptr, ck_tile::make_tuple(M, N), ck_tile::make_tuple(1, 0));
    auto const norm_y_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
      norm_y_ptr, ck_tile::make_tuple(M, N), ck_tile::make_tuple(0, 1));

    // Per-row min tracking for entire tile
    // Each thread tracks min for rows it owns
    constexpr ck_tile::index_t rows_per_thread = MPerIterationShuffle / kBlockSize + 1;
    KVPair thread_row_min[kMPerBlock];

    // Initialize all row minima to max
    for (ck_tile::index_t r = 0; r < kMPerBlock; ++r) {
      thread_row_min[r] = {0, maxVal};
    }

    constexpr ck_tile::index_t num_access = SFC::get_num_of_access();

    // Process each shuffle iteration
    ck_tile::static_for<0, num_access, 1>{}([&](auto iAccess) {
      ck_tile::block_sync_lds();
      slice_acc_tile<iAccess>(o_acc_tile, lds_tile);
      cast_lds_tile(lds_tile, in_lds_window);
      ck_tile::block_sync_lds();

      // Load accumulated values from LDS
      auto e_tensor =
        ck_tile::load_tile(ck_tile::make_tile_window(out_lds_window, dram_tile_distribution));

      constexpr auto idx_y_start          = SFC::get_index(ck_tile::number<iAccess>{});
      ck_tile::index_t const tile_m_start = i_m + idx_y_start.at(ck_tile::number<0>{});
      ck_tile::index_t const tile_n_start = i_n + idx_y_start.at(ck_tile::number<1>{});

      // Load norm tiles
      auto norm_x_tile_window =
        ck_tile::make_tile_window(norm_x_view,
                                  ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                                                      ck_tile::number<NPerIterationShuffle>{}),
                                  {tile_m_start, tile_n_start},
                                  dram_tile_distribution);
      auto norm_y_tile_window =
        ck_tile::make_tile_window(norm_y_view,
                                  ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                                                      ck_tile::number<NPerIterationShuffle>{}),
                                  {tile_m_start, tile_n_start},
                                  dram_tile_distribution);

      auto const norm_x_tile = ck_tile::load_tile(norm_x_tile_window);
      auto const norm_y_tile = ck_tile::load_tile(norm_y_tile_window);

      // Apply distance op: e = f(c, norm_x, norm_y)
      auto const op_args = ck_tile::concat_tuple_of_reference(
        ck_tile::tie(e_tensor, e_tensor), ck_tile::tie(norm_x_tile, norm_y_tile));
      ck_tile::tile_elementwise_inout_unpack(elfunc_, op_args);

      // Now e_tensor contains distances - perform per-thread reduction
      // Get thread's data and find min per row
      auto const& thread_buf                 = e_tensor.get_thread_buffer();
      ck_tile::index_t const thread_buf_size = thread_buf.size();

      // Thread layout info
      ck_tile::index_t const thread_id = threadIdx.x;

      // Compute elements per thread in M and N dimensions
      constexpr ck_tile::index_t elems_per_thread =
        MPerIterationShuffle * NPerIterationShuffle / kBlockSize;
      constexpr ck_tile::index_t threads_per_row = NPerIterationShuffle / GetVectorSizeC();
      constexpr ck_tile::index_t rows_per_iter   = MPerIterationShuffle;

      // Process thread's elements
      for (ck_tile::index_t elem = 0; elem < thread_buf_size; ++elem) {
        // Compute which row and column this element belongs to
        // Thread-raked pattern: threads are distributed across columns, each with VectorSizeC
        // elements
        ck_tile::index_t const elem_row = elem / GetVectorSizeC();
        ck_tile::index_t const elem_col_base =
          (thread_id % threads_per_row) * GetVectorSizeC() + (elem % GetVectorSizeC());
        ck_tile::index_t const local_row =
          (thread_id / threads_per_row) + elem_row * (kBlockSize / threads_per_row);

        if (local_row < MPerIterationShuffle) {
          ck_tile::index_t const global_row = idx_y_start.at(ck_tile::number<0>{}) + local_row;
          ck_tile::index_t const global_col = tile_n_start + elem_col_base;
          ck_tile::index_t const matrix_row = global_row + i_m;

          if (global_row < kMPerBlock && matrix_row < M && global_col < N) {
            ODataType dist_val = thread_buf[elem];
            KVPair candidate   = {static_cast<IdxType>(global_col), dist_val};
            thread_row_min[global_row] =
              pair_red_op_(matrix_row, candidate, thread_row_min[global_row]);
          }
        }
      }
    });

    // Warp-level reduction for rows
    ck_tile::index_t const lane_id = threadIdx.x % ck_tile::get_warp_size();
    ck_tile::index_t const warp_id = threadIdx.x / ck_tile::get_warp_size();

    // Intra-warp reduction using shuffles
    for (ck_tile::index_t r = 0; r < kMPerBlock; ++r) {
      KVPair my_val = thread_row_min[r];

      // Butterfly reduction within warp
      for (ck_tile::index_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset >>= 1) {
        IdxType other_key   = __shfl_xor(my_val.key, offset, ck_tile::get_warp_size());
        ODataType other_val = __shfl_xor(my_val.value, offset, ck_tile::get_warp_size());
        KVPair other        = {other_key, other_val};
        my_val              = pair_red_op_(r + i_m, other, my_val);
      }

      thread_row_min[r] = my_val;
    }

    // Cross-warp reduction using LDS
    ck_tile::block_sync_lds();

    // Use LDS for cross-warp communication
    KVPair* reduce_smem = reinterpret_cast<KVPair*>(p_smem);

    // Lane 0 of each warp writes its results to LDS
    if (lane_id == 0) {
      for (ck_tile::index_t r = 0; r < kMPerBlock; ++r) {
        reduce_smem[r * NWave * MWave + warp_id] = thread_row_min[r];
      }
    }

    ck_tile::block_sync_lds();

    // First warp performs final reduction across warps
    constexpr ck_tile::index_t num_warps = MWave * NWave;
    if (warp_id == 0) {
      // Pre-compute final_min for all rows this lane handles
      KVPair lane_final_min[kMPerBlock / ck_tile::get_warp_size() + 1];
      ck_tile::index_t lane_rows[kMPerBlock / ck_tile::get_warp_size() + 1];
      ck_tile::index_t num_lane_rows = 0;

      for (ck_tile::index_t r = lane_id; r < kMPerBlock; r += ck_tile::get_warp_size()) {
        KVPair final_min = reduce_smem[r * num_warps];
        for (ck_tile::index_t w = 1; w < num_warps; ++w) {
          final_min = pair_red_op_(r + i_m, reduce_smem[r * num_warps + w], final_min);
        }
        lane_final_min[num_lane_rows] = final_min;
        lane_rows[num_lane_rows]      = r;
        num_lane_rows++;
      }

      // Serialize mutex acquisition within warp to prevent hangs.
      // This resolves issues where wavefront lockstep execution causes deadlock
      // when lanes spin on different mutexes held by other blocks.
      for (ck_tile::index_t j = 0; j < ck_tile::get_warp_size(); ++j) {
        if (lane_id == j) {
          for (ck_tile::index_t idx = 0; idx < num_lane_rows; ++idx) {
            ck_tile::index_t const global_row = i_m + lane_rows[idx];
            if (global_row < M) {
              while (atomicCAS(mutex + global_row, 0, 1) == 1)
                ;
              __threadfence();
              red_op_(global_row, min_output + global_row, lane_final_min[idx]);
              __threadfence();
              atomicCAS(mutex + global_row, 1, 0);
            }
          }
        }
      }
    }
  }

 private:
  template <ck_tile::index_t iAccess, typename OAccTile, typename LdsTile>
  __device__ void slice_acc_tile(OAccTile const& o_acc_tile, LdsTile& lds_tile) const
  {
    constexpr auto idx_y_start = SFC::get_index(ck_tile::number<iAccess>{});
    constexpr auto mIter =
      ck_tile::number<idx_y_start.at(ck_tile::number<0>{}) / MPerIterationShuffle>{};
    constexpr auto nIter =
      ck_tile::number<idx_y_start.at(ck_tile::number<1>{}) / NPerIterationShuffle>{};
    constexpr auto c_warp_y_lengths =
      ck_tile::to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
      ck_tile::merge_sequences(
        ck_tile::sequence<mIter * NumMXdlPerWavePerShuffle, nIter * NumNXdlPerWavePerShuffle>{},
        c_warp_y_index_zeros),
      ck_tile::merge_sequences(
        ck_tile::sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{}, c_warp_y_lengths));
  }

  template <typename LdsTile, typename InLdsWindow>
  __device__ void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window) const
  {
    auto const c_warptile_in_tensor_casted = ck_tile::cast_tile<ODataType>(lds_tile);
    ck_tile::store_tile(in_lds_window, c_warptile_in_tensor_casted);
  }

  __host__ __device__ static constexpr auto MakeLdsBlockDescriptor()
  {
    return ck_tile::make_naive_tensor_descriptor(
      ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                          ck_tile::number<NPerIterationShuffle>{}),
      ck_tile::make_tuple(ck_tile::number<NPerIterationShuffle>{}, ck_tile::number<1>{}));
  }

  __device__ static constexpr auto MakeLdsDistributionEncode()
  {
    constexpr auto block_outer_dstr_encoding = [] {
      if constexpr (BlockedXDLN_PerWarp == 1) {
        return ck_tile::tile_distribution_encoding<
          ck_tile::sequence<>,
          ck_tile::tuple<ck_tile::sequence<NumMXdlPerWavePerShuffle, MWave>,
                         ck_tile::sequence<NumNXdlPerWavePerShuffle, NWave>>,
          ck_tile::tuple<ck_tile::sequence<1, 2>>,
          ck_tile::tuple<ck_tile::sequence<1, 1>>,
          ck_tile::sequence<1, 2>,
          ck_tile::sequence<0, 0>>{};
      } else {
        constexpr int RakedXDLN_PerWarp = NumNXdlPerWavePerShuffle / BlockedXDLN_PerWarp;
        return ck_tile::tile_distribution_encoding<
          ck_tile::sequence<>,
          ck_tile::tuple<ck_tile::sequence<NumMXdlPerWavePerShuffle, MWave>,
                         ck_tile::sequence<RakedXDLN_PerWarp, NWave, BlockedXDLN_PerWarp>>,
          ck_tile::tuple<ck_tile::sequence<1, 2>>,
          ck_tile::tuple<ck_tile::sequence<1, 1>>,
          ck_tile::sequence<1, 2, 2>,
          ck_tile::sequence<0, 0, 2>>{};
      }
    }();
    return ck_tile::detail::make_embed_tile_distribution_encoding(block_outer_dstr_encoding,
                                                                  typename CWarpDstr::DstrEncode{});
  }
};

}  // namespace cuvs::distance::detail
