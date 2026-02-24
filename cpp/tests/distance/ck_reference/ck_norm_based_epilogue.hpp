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

// Generic norm-based epilogue for CK Tile GEMM: loads norm_x (m) and norm_y (n)
// per tile, then applies ElementWiseOp(e, c, norm_x, norm_y). Supports
// CosineDistance, L2Expanded, and other metrics without materializing the m×n
// D0 tensor.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/core/tensor/tile_distribution_encoding.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <ck_tile/ops/common/utils.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>

namespace ck_tile {

// Problem config matching CShuffleEpilogue layout (TiledMMAPermuteN=false).
template <typename AccDataType_,
          typename ODataType_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool FixedVectorSize_        = false,
          index_t VectorSizeC_         = 1,
          index_t BlockedXDLN_PerWarp_ = 1>
struct PairwiseDistanceCkEpilogueProblem {
  using AccDataType = remove_cvref_t<AccDataType_>;
  using ODataType   = remove_cvref_t<ODataType_>;

  static constexpr index_t kBlockSize          = MWave_ * NWave_ * get_warp_size();
  static constexpr index_t kMPerBlock          = kM_;
  static constexpr index_t kNPerBlock          = kN_;
  static constexpr index_t MWave               = MWave_;
  static constexpr index_t NWave               = NWave_;
  static constexpr index_t MPerXdl             = MPerXdl_;
  static constexpr index_t NPerXdl             = NPerXdl_;
  static constexpr index_t KPerXdl             = KPerXdl_;
  static constexpr bool FixedVectorSize        = FixedVectorSize_;
  static constexpr index_t VectorSizeC         = VectorSizeC_;
  static constexpr index_t BlockedXDLN_PerWarp = BlockedXDLN_PerWarp_;

  static constexpr index_t MPerIteration = MPerXdl * MWave;
  static constexpr index_t NPerIteration = NPerXdl * NWave;
  static constexpr index_t MRepeat       = kMPerBlock / (MPerXdl * MWave);
  static constexpr index_t NRepeat       = kNPerBlock / (NPerXdl * NWave);

  using ELayout = tensor_layout::gemm::RowMajor;
};

// Generic epilogue: ElementWiseOp(e, c, norm_x, norm_y).
template <typename Problem_, typename ElementWiseOp_>
struct PairwiseDistanceCkEpilogue {
  using Problem       = remove_cvref_t<Problem_>;
  using ElementWiseOp = remove_cvref_t<ElementWiseOp_>;
  using AccDataType   = remove_cvref_t<typename Problem::AccDataType>;
  using ODataType     = remove_cvref_t<typename Problem::ODataType>;

  static constexpr index_t kBlockSize                    = Problem::kBlockSize;
  static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
  static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;
  static constexpr index_t MWave                         = Problem::MWave;
  static constexpr index_t NWave                         = Problem::NWave;
  static constexpr index_t MPerXdl                       = Problem::MPerXdl;
  static constexpr index_t NPerXdl                       = Problem::NPerXdl;
  static constexpr index_t MPerIteration                 = Problem::MPerIteration;
  static constexpr index_t NPerIteration                 = Problem::NPerIteration;
  static constexpr index_t MRepeat                       = Problem::MRepeat;
  static constexpr index_t NRepeat                       = Problem::NRepeat;
  static constexpr index_t BlockedXDLN_PerWarp           = Problem::BlockedXDLN_PerWarp;
  static constexpr memory_operation_enum MemoryOperation = memory_operation_enum::set;

  using ATypeToUse = ODataType;
  using BTypeToUse = ODataType;
  using WG         = WarpGemmDispatcher<ATypeToUse,
                                        BTypeToUse,
                                        AccDataType,
                                        MPerXdl,
                                        NPerXdl,
                                        Problem::KPerXdl,
                                        false>;

  using CWarpDstr   = typename WG::CWarpDstr;
  using CWarpTensor = typename WG::CWarpTensor;

  static constexpr index_t GetVectorSizeC()
  {
    if constexpr (Problem::FixedVectorSize) return Problem::VectorSizeC;
    constexpr index_t max_vector_size = 16;
    return std::min(static_cast<int>(NPerIteration),
                    static_cast<int>(max_vector_size / sizeof(ODataType)));
  }

  static constexpr auto shuffle_tile_tuple = [] {
    constexpr index_t elem_per_thread = MPerXdl * NPerXdl / get_warp_size();
    constexpr index_t max_vector_size = 16;
    if constexpr (elem_per_thread >= GetVectorSizeC()) {
      return std::make_tuple(1, 1);
    } else {
      constexpr index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
      return std::make_tuple(min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
    }
  }();
  static constexpr index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
  static constexpr index_t NumNXdlPerWavePerShuffle =
    max(BlockedXDLN_PerWarp, std::get<1>(shuffle_tile_tuple));

  static constexpr auto MNPerIterationShuffle = [] {
    constexpr index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;
    constexpr index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;
    if constexpr (kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
      return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);
    else
      return std::make_tuple(m_val, n_val);
  }();
  static constexpr index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);
  static constexpr index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);

  using SFC = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                  sequence<0, 1>,
                                  sequence<MPerIterationShuffle, NPerIterationShuffle>>;

  __host__ __device__ static constexpr index_t GetSmemSize()
  {
    return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
  }

  ElementWiseOp elfunc_;

  __device__ PairwiseDistanceCkEpilogue(ElementWiseOp elfunc = {}) : elfunc_(elfunc) {}

  template <typename ODramWindow, typename OAccTile>
  __device__ void operator()(ODramWindow& out_dram_window,
                             const OAccTile& o_acc_tile,
                             void* p_smem,
                             const ODataType* norm_x_ptr,
                             const ODataType* norm_y_ptr,
                             index_t i_m,
                             index_t i_n,
                             index_t M,
                             index_t N) const
  {
    using TileEncodingPattern =
      tile_distribution_encoding_pattern_2d<kBlockSize,
                                            MPerIterationShuffle,
                                            NPerIterationShuffle,
                                            GetVectorSizeC(),
                                            tile_distribution_pattern::thread_raked,
                                            1>;
    constexpr auto LdsTileDistr = make_static_tile_distribution(MakeLdsDistributionEncode());
    constexpr auto dram_tile_distribution = TileEncodingPattern::make_2d_static_tile_distribution();

    constexpr auto lds_block_desc = MakeLdsBlockDescriptor();
    auto o_lds_block =
      make_tensor_view<address_space_enum::lds>(static_cast<ODataType*>(p_smem), lds_block_desc);

    auto lds_tile = make_static_distributed_tensor<AccDataType>(LdsTileDistr);

    auto in_lds_window =
      make_tile_window(o_lds_block,
                       make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                       {0, 0},
                       LdsTileDistr);

    auto out_lds_window =
      make_tile_window(o_lds_block,
                       make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                       {0, 0});

    // norm_x: shape (M,N) with stride (1,0) -> broadcast norm_x[i] over j
    // norm_y: shape (M,N) with stride (0,1) -> broadcast norm_y[j] over i
    const auto norm_x_view = make_naive_tensor_view<address_space_enum::global>(
      norm_x_ptr, make_tuple(M, N), make_tuple(1, 0));
    const auto norm_y_view = make_naive_tensor_view<address_space_enum::global>(
      norm_y_ptr, make_tuple(M, N), make_tuple(0, 1));

    constexpr index_t num_access = SFC::get_num_of_access();

    static_for<0, num_access, 1>{}([&](auto iAccess) {
      block_sync_lds();
      slice_acc_tile<iAccess>(o_acc_tile, lds_tile);
      cast_lds_tile(lds_tile, in_lds_window);
      block_sync_lds();

      auto e_tensor = load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

      constexpr auto idx_y_start = SFC::get_index(number<iAccess>{});
      const index_t tile_m_start = i_m + idx_y_start.at(number<0>{});
      const index_t tile_n_start = i_n + idx_y_start.at(number<1>{});

      auto norm_x_tile_window =
        make_tile_window(norm_x_view,
                         make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                         {tile_m_start, tile_n_start},
                         dram_tile_distribution);
      auto norm_y_tile_window =
        make_tile_window(norm_y_view,
                         make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                         {tile_m_start, tile_n_start},
                         dram_tile_distribution);

      const auto norm_x_tile = load_tile(norm_x_tile_window);
      const auto norm_y_tile = load_tile(norm_y_tile_window);

      // ElementWiseOp(e, c, norm_x, norm_y): e and c both reference e_tensor (c = dot from LDS)
      const auto op_args =
        concat_tuple_of_reference(tie(e_tensor, e_tensor), tie(norm_x_tile, norm_y_tile));
      tile_elementwise_inout_unpack(elfunc_, op_args);

      if constexpr (MemoryOperation == memory_operation_enum::set) {
        store_tile(out_dram_window, e_tensor);
      } else {
        update_tile(out_dram_window, e_tensor);
      }

      if constexpr (iAccess != num_access - 1) {
        constexpr auto step = SFC::get_forward_step(number<iAccess>{});
        move_tile_window(out_dram_window, {step.at(number<0>{}), step.at(number<1>{})});
      }
    });
  }

 private:
  template <index_t iAccess, typename OAccTile, typename LdsTile>
  __device__ void slice_acc_tile(const OAccTile& o_acc_tile, LdsTile& lds_tile) const
  {
    constexpr auto idx_y_start = SFC::get_index(number<iAccess>{});
    constexpr auto mIter       = number<idx_y_start.at(number<0>{}) / MPerIterationShuffle>{};
    constexpr auto nIter       = number<idx_y_start.at(number<1>{}) / NPerIterationShuffle>{};
    constexpr auto c_warp_y_lengths =
      to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
      merge_sequences(
        sequence<mIter * NumMXdlPerWavePerShuffle, nIter * NumNXdlPerWavePerShuffle>{},
        c_warp_y_index_zeros),
      merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                      c_warp_y_lengths));
  }

  template <typename LdsTile, typename InLdsWindow>
  __device__ void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window) const
  {
    const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile);
    store_tile(in_lds_window, c_warptile_in_tensor_casted);
  }

  __host__ __device__ static constexpr auto MakeLdsBlockDescriptor()
  {
    return make_naive_tensor_descriptor(
      make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
      make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
  }

  __device__ static constexpr auto MakeLdsDistributionEncode()
  {
    constexpr auto block_outer_dstr_encoding = [] {
      if constexpr (BlockedXDLN_PerWarp == 1) {
        return tile_distribution_encoding<sequence<>,
                                          tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                                sequence<NumNXdlPerWavePerShuffle, NWave>>,
                                          tuple<sequence<1, 2>>,
                                          tuple<sequence<1, 1>>,
                                          sequence<1, 2>,
                                          sequence<0, 0>>{};
      } else {
        constexpr int RakedXDLN_PerWarp = NumNXdlPerWavePerShuffle / BlockedXDLN_PerWarp;
        return tile_distribution_encoding<
          sequence<>,
          tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                sequence<RakedXDLN_PerWarp, NWave, BlockedXDLN_PerWarp>>,
          tuple<sequence<1, 2>>,
          tuple<sequence<1, 1>>,
          sequence<1, 2, 2>,
          sequence<0, 0, 2>>{};
      }
    }();
    constexpr auto block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
      block_outer_dstr_encoding, typename CWarpDstr::DstrEncode{});
    return block_dstr_encoding;
  }
};

}  // namespace ck_tile
