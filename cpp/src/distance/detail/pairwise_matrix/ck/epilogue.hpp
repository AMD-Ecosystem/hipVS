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

// Pairwise distance epilogue for CK Tile GEMM: loads norm_x (m) and norm_y (n)
// per tile, then applies ElementWiseOp(e, c, norm_x, norm_y). Supports
// CosineDistance, L2Expanded, and other metrics without materializing the m×n
// D0 tensor. Folds the thrust::tabulate computation into the kernel.
//
// --- Why this custom epilogue is required ---
// CK's CShuffleEpilogue (ck_tile/ops/epilogue/cshuffle_epilogue.hpp) expects
// D0, D1, ... as full m×n tensors (DsDramWindows) passed to operator(). The
// element-wise op receives (c_out_tensor, d0_tile, d1_tile, ...). For pairwise
// distance we need norm_x[i] (length m) and norm_y[j] (length n) broadcast over
// the tile to compute e.g. cosine(e,c,nx,ny)=1-c/(nx*ny) or L2_expanded without
// materializing an m×n D0. CShuffleEpilogue cannot express this 1D→2D broadcast,
// so we provide a custom epilogue that loads norm tiles from vectors.
//
// --- Relationship to CK CShuffleEpilogue ---
// This mirrors CShuffleEpilogue with TiledMMAPermuteN=false (the non-permute
// code path). We reuse the same tile layout and LDS shuffle so the GEMM
// pipeline output format remains compatible.
//
// 1:1 IDENTICAL to CShuffleEpilogue (TiledMMAPermuteN=false):
//   - PairwiseDistanceCkEpilogueProblem: kM, kN, MWave, NWave, MPerXdl, NPerXdl,
//     KPerXdl, BlockedXDLN_PerWarp, FixedVectorSize, VectorSizeC, ELayout
//   - shuffle_tile_tuple (elem_per_thread, num_xdl_shuffles for RowMajor)
//   - NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle
//   - MNPerIterationShuffle, MPerIterationShuffle, NPerIterationShuffle
//   - SFC (space_filling_curve), TileEncodingPattern, dram_tile_distribution
//   - MakeLdsBlockDescriptor (RowMajor), MakeLdsDistributionEncode
//   - slice_acc_tile, cast_lds_tile
//   - Loop: block_sync_lds, slice_acc_tile, cast_lds_tile, block_sync_lds,
//     load_tile, store_tile/update_tile, move_tile_window
//
// MODIFIED for pairwise distance:
//   - operator() signature: norm_x_ptr, norm_y_ptr, i_m, i_n, M, N instead of
//     ds_dram_windows (no D tensors). No ScaleM/ScaleN.
//   - norm_x_view, norm_y_view: 1D vectors with strides (1,0) and (0,1) to
//     form logical (M,N) broadcast views (norm_x[i] over j, norm_y[j] over i).
//   - norm_x_tile_window, norm_y_tile_window: tile windows at (tile_m_start,
//     tile_n_start) to load per-tile norm tiles.
//   - ElementWiseOp(e, c, norm_x, norm_y): our distance op instead of
//     CShuffleEpilogue's apply_d_tensors which uses op(e, c, d0, d1, ...).

#pragma once

#include <algorithm>
#include <ck_tile/core.hpp>
#include <ck_tile/core/tensor/tile_distribution_encoding.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <ck_tile/ops/common/utils.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>

namespace ck {

// --- Problem config: tile layout matching CShuffleEpilogue (TiledMMAPermuteN=false) ---
// [CShuffleEpilogue] Problem struct; we omit DsDataType, DsLayout, NumDTensor (no D tensors)
template <typename AccDataType_,
          typename ODataType_,
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
struct PairwiseDistanceCkEpilogueProblem {
  using AccDataType = ck_tile::remove_cvref_t<AccDataType_>;
  using ODataType   = ck_tile::remove_cvref_t<ODataType_>;

  // [CShuffleEpilogue] Same layout constants
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

// --- Epilogue: loads norm_x/norm_y per tile, applies ElementWiseOp(e, c, norm_x, norm_y) ---
template <typename Problem_, typename ElementWiseOp_>
struct PairwiseDistanceCkEpilogue {
  using Problem       = ck_tile::remove_cvref_t<Problem_>;
  using ElementWiseOp = ck_tile::remove_cvref_t<ElementWiseOp_>;
  using AccDataType   = ck_tile::remove_cvref_t<typename Problem::AccDataType>;
  using ODataType     = ck_tile::remove_cvref_t<typename Problem::ODataType>;

  // [CShuffleEpilogue] Same static config from Problem
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
  static constexpr ck_tile::memory_operation_enum MemoryOperation =
    ck_tile::memory_operation_enum::set;

  // [CShuffleEpilogue] WG, CWarpDstr, CWarpTensor (isCTransposed=false for RowMajor)
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

  // [CShuffleEpilogue] Same GetVectorSizeC for RowMajor
  static constexpr ck_tile::index_t GetVectorSizeC()
  {
    if constexpr (Problem::FixedVectorSize) return Problem::VectorSizeC;
    constexpr ck_tile::index_t max_vector_size = 16;
    return std::min(static_cast<int>(NPerIteration),
                    static_cast<int>(max_vector_size / sizeof(ODataType)));
  }

  // [CShuffleEpilogue] Same shuffle_tile_tuple logic (RowMajor path)
  static constexpr auto shuffle_tile_tuple = [] {
    constexpr ck_tile::index_t elem_per_thread = MPerXdl * NPerXdl / ck_tile::get_warp_size();
    constexpr ck_tile::index_t max_vector_size = 16;
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

  // [CShuffleEpilogue] Same MNPerIterationShuffle
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

  // [CShuffleEpilogue] Same SFC (space_filling_curve)
  using SFC =
    ck_tile::space_filling_curve<ck_tile::sequence<kMPerBlock, kNPerBlock>,
                                 ck_tile::sequence<0, 1>,
                                 ck_tile::sequence<MPerIterationShuffle, NPerIterationShuffle>>;

  // [CShuffleEpilogue] Same GetSmemSize
  __host__ __device__ static constexpr ck_tile::index_t GetSmemSize()
  {
    return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
  }

  ElementWiseOp elfunc_;

  __device__ PairwiseDistanceCkEpilogue(ElementWiseOp elfunc = {}) : elfunc_(elfunc) {}

  // [MODIFIED] Signature: norm_x_ptr, norm_y_ptr, i_m, i_n, M, N instead of ds_dram_windows
  template <typename ODramWindow, typename OAccTile>
  __device__ void operator()(ODramWindow& out_dram_window,
                             const OAccTile& o_acc_tile,
                             void* p_smem,
                             const ODataType* norm_x_ptr,
                             const ODataType* norm_y_ptr,
                             ck_tile::index_t i_m,
                             ck_tile::index_t i_n,
                             ck_tile::index_t M,
                             ck_tile::index_t N) const
  {
    // [CShuffleEpilogue] Same TileEncodingPattern, LdsTileDistr, dram_tile_distribution
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

    // [CShuffleEpilogue] Same LDS setup (lds_block_desc, o_lds_block, in/out_lds_window)
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

    // [MODIFIED] 1D norm vectors as (M,N) views with broadcast strides (1,0) and (0,1)
    const auto norm_x_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
      norm_x_ptr, ck_tile::make_tuple(M, N), ck_tile::make_tuple(1, 0));
    const auto norm_y_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
      norm_y_ptr, ck_tile::make_tuple(M, N), ck_tile::make_tuple(0, 1));

    constexpr ck_tile::index_t num_access = SFC::get_num_of_access();

    ck_tile::static_for<0, num_access, 1>{}([&](auto iAccess) {
      // [CShuffleEpilogue] Same: block_sync, slice_acc_tile, cast_lds_tile, block_sync
      ck_tile::block_sync_lds();
      slice_acc_tile<iAccess>(o_acc_tile, lds_tile);
      cast_lds_tile(lds_tile, in_lds_window);
      ck_tile::block_sync_lds();

      // [CShuffleEpilogue] Same: load c_out_tensor from LDS
      auto e_tensor =
        ck_tile::load_tile(ck_tile::make_tile_window(out_lds_window, dram_tile_distribution));

      constexpr auto idx_y_start          = SFC::get_index(ck_tile::number<iAccess>{});
      const ck_tile::index_t tile_m_start = i_m + idx_y_start.at(ck_tile::number<0>{});
      const ck_tile::index_t tile_n_start = i_n + idx_y_start.at(ck_tile::number<1>{});

      // [MODIFIED] Load norm tiles from 1D vectors; CShuffleEpilogue uses d_dram_windows
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

      const auto norm_x_tile = ck_tile::load_tile(norm_x_tile_window);
      const auto norm_y_tile = ck_tile::load_tile(norm_y_tile_window);

      // [MODIFIED] op(e, c, norm_x, norm_y) instead of apply_d_tensors(op, c, d0, d1, ...)
      const auto op_args = ck_tile::concat_tuple_of_reference(
        ck_tile::tie(e_tensor, e_tensor), ck_tile::tie(norm_x_tile, norm_y_tile));
      ck_tile::tile_elementwise_inout_unpack(elfunc_, op_args);

      // [CShuffleEpilogue] Same: store_to_dram (store_tile/update_tile)
      if constexpr (MemoryOperation == ck_tile::memory_operation_enum::set) {
        ck_tile::store_tile(out_dram_window, e_tensor);
      } else {
        ck_tile::update_tile(out_dram_window, e_tensor);
      }

      // [CShuffleEpilogue] Same: move_tile_window (we only move output; no D windows)
      if constexpr (iAccess != num_access - 1) {
        constexpr auto step = SFC::get_forward_step(ck_tile::number<iAccess>{});
        ck_tile::move_tile_window(out_dram_window,
                                  {step.at(ck_tile::number<0>{}), step.at(ck_tile::number<1>{})});
      }
    });
  }

 private:
  // [CShuffleEpilogue] Same slice_acc_tile
  template <ck_tile::index_t iAccess, typename OAccTile, typename LdsTile>
  __device__ void slice_acc_tile(const OAccTile& o_acc_tile, LdsTile& lds_tile) const
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

  // [CShuffleEpilogue] Same cast_lds_tile
  template <typename LdsTile, typename InLdsWindow>
  __device__ void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window) const
  {
    const auto c_warptile_in_tensor_casted = ck_tile::cast_tile<ODataType>(lds_tile);
    ck_tile::store_tile(in_lds_window, c_warptile_in_tensor_casted);
  }

  // [CShuffleEpilogue] Same MakeLdsBlockDescriptor (RowMajor)
  __host__ __device__ static constexpr auto MakeLdsBlockDescriptor()
  {
    return ck_tile::make_naive_tensor_descriptor(
      ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                          ck_tile::number<NPerIterationShuffle>{}),
      ck_tile::make_tuple(ck_tile::number<NPerIterationShuffle>{}, ck_tile::number<1>{}));
  }

  // [CShuffleEpilogue] Same MakeLdsDistributionEncode
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

}  // namespace ck
