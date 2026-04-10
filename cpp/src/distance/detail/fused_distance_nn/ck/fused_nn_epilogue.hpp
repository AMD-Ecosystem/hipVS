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

  // Main operator: transforms GEMM dot-product accumulator into per-row nearest-neighbor results.
  //
  // Pipeline overview:
  //   1. Distance computation: MFMA layout -> LDS shuffle -> thread-raked layout -> elfunc_
  //   2. Reduction: intra-thread -> intra-warp shuffle -> cross-warp LDS -> atomic global update
  //
  // Parameters:
  //   min_output  - [M] global output: KeyValuePair per row tracking (nearest_col, min_distance)
  //   mutex       - [M] per-row spinlock (only used in the fallback path for >8-byte OutT)
  //   o_acc_tile  - GEMM accumulator tile: C[i][j] = dot(X[i_m+i,:], Y[i_n+j,:]) in MFMA layout
  //   p_smem      - LDS workspace, reused for tile shuffle (phase 1) and cross-warp reduction
  //   (phase 2) norm_x_ptr  - [M] pre-computed row norms of X (||x_i||^2 for L2, ||x_i|| for
  //   cosine) norm_y_ptr  - [N] pre-computed row norms of Y i_m, i_n    - tile origin in the global
  //   M x N grid M, N        - full matrix dimensions (for bounds checking) maxVal      -
  //   initialization sentinel for the reduction (e.g., +inf)
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
    // ===========================================================================================
    // Setup: tile distribution encodings and LDS views
    // ===========================================================================================

    // Thread-raked distribution for reading back from LDS: each thread gets VectorSizeC
    // contiguous columns, threads tile across columns first then rows.
    // E.g., with VectorSizeC=4, threads_per_row=8: Thread 0 -> row 0 cols [0..3],
    // Thread 1 -> row 0 cols [4..7], ..., Thread 8 -> row 1 cols [0..3], etc.
    using TileEncodingPattern = ck_tile::tile_distribution_encoding_pattern_2d<
      kBlockSize,
      MPerIterationShuffle,
      NPerIterationShuffle,
      GetVectorSizeC(),
      ck_tile::tile_distribution_pattern::thread_raked,
      1>;

    // MFMA distribution for writing TO LDS: maps each thread's MFMA register positions to
    // the corresponding row-major LDS locations, undoing the hardware's scattered layout.
    constexpr auto LdsTileDistr =
      ck_tile::make_static_tile_distribution(MakeLdsDistributionEncode());

    // Thread-raked distribution for reading FROM LDS after the layout transformation.
    constexpr auto dram_tile_distribution = TileEncodingPattern::make_2d_static_tile_distribution();

    // LDS viewed as a row-major 2D tensor: [MPerIterationShuffle x NPerIterationShuffle]
    constexpr auto lds_block_desc = MakeLdsBlockDescriptor();
    auto o_lds_block              = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
      static_cast<ODataType*>(p_smem), lds_block_desc);

    // Register buffer to hold the MFMA sub-tile slice before writing to LDS.
    auto lds_tile = ck_tile::make_static_distributed_tensor<AccDataType>(LdsTileDistr);

    // Write window: stores data from MFMA layout into LDS (uses LdsTileDistr to scatter-write).
    auto in_lds_window =
      ck_tile::make_tile_window(o_lds_block,
                                ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                                                    ck_tile::number<NPerIterationShuffle>{}),
                                {0, 0},
                                LdsTileDistr);

    // Read window: loads data from LDS using thread-raked pattern (no distribution attached,
    // dram_tile_distribution is applied at load_tile time to determine each thread's read offsets).
    auto out_lds_window =
      ck_tile::make_tile_window(o_lds_block,
                                ck_tile::make_tuple(ck_tile::number<MPerIterationShuffle>{},
                                                    ck_tile::number<NPerIterationShuffle>{}),
                                {0, 0});

    // Broadcast views for the 1D norm vectors:
    //   norm_x: shape (M, N) with stride (1, 0) -> norm_x[i][j] = norm_x_ptr[i] for all j
    //   norm_y: shape (M, N) with stride (0, 1) -> norm_y[i][j] = norm_y_ptr[j] for all i
    // The stride-0 dimension causes the hardware to replicate the value across that axis
    // without extra memory reads, effectively broadcasting a 1D vector into a 2D tile.
    auto const norm_x_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
      norm_x_ptr, ck_tile::make_tuple(M, N), ck_tile::make_tuple(1, 0));
    auto const norm_y_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
      norm_y_ptr, ck_tile::make_tuple(M, N), ck_tile::make_tuple(0, 1));

    // ===========================================================================================
    // Reduction state: per-row minimum tracking across all sub-tile iterations
    // ===========================================================================================

    // Each thread maintains a running min KVPair for every row in the block tile.
    // Most entries stay at maxVal (only rows the thread actually touches get updated).
    // This array persists across all SFC iterations, accumulating the thread-local
    // minimum across the full kNPerBlock column range.
    constexpr ck_tile::index_t rows_per_thread = MPerIterationShuffle / kBlockSize + 1;
    KVPair thread_row_min[kMPerBlock];

    for (ck_tile::index_t r = 0; r < kMPerBlock; ++r) {
      thread_row_min[r] = {0, maxVal};
    }

    // ===========================================================================================
    // Phase 1: Distance computation + intra-thread reduction
    //
    // The space-filling curve (SFC) breaks the full kMPerBlock x kNPerBlock tile into
    // sub-tiles of size MPerIterationShuffle x NPerIterationShuffle. For each sub-tile:
    //   a) Shuffle GEMM accumulator from MFMA registers -> LDS -> thread-raked registers
    //   b) Load broadcast norms and apply element-wise distance op
    //   c) Each thread reduces its distance elements into thread_row_min[]
    // ===========================================================================================

    constexpr ck_tile::index_t num_access = SFC::get_num_of_access();

    ck_tile::static_for<0, num_access, 1>{}([&](auto iAccess) {
      // --- Stage 1a: LDS layout shuffle (MFMA -> row-major -> thread-raked) ---

      // Ensure previous iteration's LDS reads are complete before we overwrite.
      ck_tile::block_sync_lds();

      // Extract the sub-tile slice from the full MFMA accumulator into lds_tile's register
      // buffer. This is a register-to-register copy; no memory movement yet.
      slice_acc_tile<iAccess>(o_acc_tile, lds_tile);

      // Cast AccDataType -> ODataType and scatter-write to LDS in row-major order.
      // Each thread writes its MFMA-distributed elements to the LDS positions determined
      // by MakeLdsDistributionEncode(), transforming the scattered MFMA layout into a
      // simple row-major 2D array in shared memory.
      cast_lds_tile(lds_tile, in_lds_window);

      // Wait for all threads to finish writing before reading back in a different layout.
      ck_tile::block_sync_lds();

      // Read from LDS using the thread-raked distribution: each thread loads VectorSizeC
      // contiguous columns for its assigned row(s). After this, e_tensor has the same data
      // as the MFMA accumulator but in a layout where each thread owns contiguous column
      // elements -- ideal for element-wise ops and per-row reduction.
      auto e_tensor =
        ck_tile::load_tile(ck_tile::make_tile_window(out_lds_window, dram_tile_distribution));

      // --- Stage 1b: Distance computation ---

      // Sub-tile origin within the block tile and within the global matrix.
      constexpr auto idx_y_start          = SFC::get_index(ck_tile::number<iAccess>{});
      ck_tile::index_t const tile_m_start = i_m + idx_y_start.at(ck_tile::number<0>{});
      ck_tile::index_t const tile_n_start = i_n + idx_y_start.at(ck_tile::number<1>{});

      // Create tile windows over the broadcast norm views, positioned at the sub-tile's
      // global coordinates. These use the same thread-raked distribution as e_tensor,
      // so each thread's norm values correspond 1:1 with its distance elements.
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

      // Apply the element-wise distance op: elfunc_(e, c, norm_x, norm_y).
      // elfunc_ is an OperatorWrapper that chains:
      //   1. Distance op:  e.g. L2: e = clamp(nx + ny - 2c)  or  Cosine: e = 1 - c/(nx*ny)
      //   2. Optional sqrt: e = sqrt(e)  (compile-time flag)
      //   3. Final op:     e = fin_op(e)  (identity in the fused NN path)
      // After this, e_tensor contains actual distance values (not dot products).
      auto const op_args = ck_tile::concat_tuple_of_reference(
        ck_tile::tie(e_tensor, e_tensor), ck_tile::tie(norm_x_tile, norm_y_tile));
      ck_tile::tile_elementwise_inout_unpack(elfunc_, op_args);

      // --- Stage 1c: Intra-thread reduction (Level 1) ---
      //
      // Each thread iterates over its elements in the thread-raked buffer, maps each to
      // its (global_row, global_col) coordinate, and reduces against thread_row_min[].
      //
      // Thread-raked coordinate mapping:
      //   threads_per_row = NPerIterationShuffle / VectorSizeC
      //   col = (thread_id % threads_per_row) * VectorSizeC + (elem % VectorSizeC)
      //   row = (thread_id / threads_per_row) + elem_row * (kBlockSize / threads_per_row)
      //
      // Example with defaults (256 threads, VectorSizeC=4, NPerIterationShuffle=32):
      //   threads_per_row = 8, so Thread 0 -> row 0 cols [0..3], Thread 7 -> row 0 cols [28..31]
      //   Thread 8 -> row 1 cols [0..3], etc.

      auto const& thread_buf                 = e_tensor.get_thread_buffer();
      ck_tile::index_t const thread_buf_size = thread_buf.size();

      ck_tile::index_t const thread_id = threadIdx.x;

      constexpr ck_tile::index_t elems_per_thread =
        MPerIterationShuffle * NPerIterationShuffle / kBlockSize;
      constexpr ck_tile::index_t threads_per_row = NPerIterationShuffle / GetVectorSizeC();
      constexpr ck_tile::index_t rows_per_iter   = MPerIterationShuffle;

      for (ck_tile::index_t elem = 0; elem < thread_buf_size; ++elem) {
        // Map this element's position in the thread buffer to its (row, col) in the sub-tile.
        ck_tile::index_t const elem_row = elem / GetVectorSizeC();
        ck_tile::index_t const elem_col_base =
          (thread_id % threads_per_row) * GetVectorSizeC() + (elem % GetVectorSizeC());
        ck_tile::index_t const local_row =
          (thread_id / threads_per_row) + elem_row * (kBlockSize / threads_per_row);

        if (local_row < MPerIterationShuffle) {
          // Translate sub-tile-local row to block-tile row and global matrix row.
          ck_tile::index_t const global_row = idx_y_start.at(ck_tile::number<0>{}) + local_row;
          ck_tile::index_t const global_col = tile_n_start + elem_col_base;
          ck_tile::index_t const matrix_row = global_row + i_m;

          if (global_row < kMPerBlock && matrix_row < M && global_col < N) {
            // Construct a (column_index, distance) candidate and keep the minimum per row.
            // pair_red_op_ returns the KVPair with the smaller .value field.
            ODataType dist_val = thread_buf[elem];
            KVPair candidate   = {static_cast<IdxType>(global_col), dist_val};
            thread_row_min[global_row] =
              pair_red_op_(matrix_row, candidate, thread_row_min[global_row]);
          }
        }
      }
    });
    // After this loop, each thread's thread_row_min[r] holds the minimum (col, dist) across
    // all N-columns that this thread processed for block-tile row r. Most entries are still
    // maxVal for rows the thread did not own elements for.

    // ===========================================================================================
    // Phase 2: Hierarchical cross-thread reduction
    //
    // Collapse thread_row_min[] from all threads down to a single KVPair per row,
    // then atomically update the global output.
    // ===========================================================================================

    ck_tile::index_t const lane_id = threadIdx.x % ck_tile::get_warp_size();
    ck_tile::index_t const warp_id = threadIdx.x / ck_tile::get_warp_size();

    // --- Level 2: Intra-warp reduction (butterfly shuffle) ---
    //
    // For each row, perform a butterfly reduction across all lanes in the warp using
    // __shfl_xor. This takes 6 rounds for warp_size=64 (log2(64) = 6), exchanging
    // key and value separately since __shfl_xor operates on scalar types.
    // After all rounds, lane 0 holds the warp-minimum for every row.
    // All lanes participate symmetrically but only lane 0's result is used.
    for (ck_tile::index_t r = 0; r < kMPerBlock; ++r) {
      KVPair my_val = thread_row_min[r];

      for (ck_tile::index_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset >>= 1) {
        IdxType other_key   = __shfl_xor(my_val.key, offset, ck_tile::get_warp_size());
        ODataType other_val = __shfl_xor(my_val.value, offset, ck_tile::get_warp_size());
        KVPair other        = {other_key, other_val};
        my_val              = pair_red_op_(r + i_m, other, my_val);
      }

      thread_row_min[r] = my_val;
    }

    // --- Level 3: Cross-warp reduction (LDS exchange) ---
    //
    // Shuffles can only communicate within a single warp. To reduce across warps
    // (num_warps = MWave * NWave, e.g. 4), we use LDS as an intermediary:
    //   1. Lane 0 of each warp writes its per-row minima to LDS
    //   2. Barrier
    //   3. Warp 0 reads all warps' results and reduces across them
    //
    // LDS layout: [kMPerBlock x num_warps] array of KVPair
    //   reduce_smem[row * num_warps + warp_id] = warp's minimum for that row
    ck_tile::block_sync_lds();

    // Reuse the same LDS that was used for the tile shuffle (the shuffle is done;
    // this region is now available). The LDS is sized as max(shuffle_size, reduce_size).
    KVPair* reduce_smem = reinterpret_cast<KVPair*>(p_smem);

    // Lane 0 of each warp writes its per-row results.
    // Only lane 0 has the correct reduced values (from the butterfly reduction above).
    if (lane_id == 0) {
      for (ck_tile::index_t r = 0; r < kMPerBlock; ++r) {
        reduce_smem[r * NWave * MWave + warp_id] = thread_row_min[r];
      }
    }

    // Ensure all warps have written before warp 0 reads.
    ck_tile::block_sync_lds();

    // Only warp 0 participates from here: reduce across warps, then update global output.
    constexpr ck_tile::index_t num_warps = MWave * NWave;
    if (warp_id == 0) {
      // Distribute rows across warp 0's lanes in a strided pattern:
      //   lane 0 -> rows {0, 64, 128, ...}
      //   lane 1 -> rows {1, 65, 129, ...}
      //   ...
      // With kMPerBlock=64 and warp_size=64, each lane handles exactly 1 row.
      KVPair lane_final_min[kMPerBlock / ck_tile::get_warp_size() + 1];
      ck_tile::index_t lane_rows[kMPerBlock / ck_tile::get_warp_size() + 1];
      ck_tile::index_t num_lane_rows = 0;

      for (ck_tile::index_t r = lane_id; r < kMPerBlock; r += ck_tile::get_warp_size()) {
        // Sequentially reduce across all warps' entries for this row.
        KVPair final_min = reduce_smem[r * num_warps];
        for (ck_tile::index_t w = 1; w < num_warps; ++w) {
          final_min = pair_red_op_(r + i_m, reduce_smem[r * num_warps + w], final_min);
        }
        lane_final_min[num_lane_rows] = final_min;
        lane_rows[num_lane_rows]      = r;
        num_lane_rows++;
      }
      // At this point, each lane in warp 0 holds the block-level minimum for its
      // assigned row(s). lane_final_min[idx] is the minimum (col, dist) across ALL
      // threads and warps in this block for row lane_rows[idx].

      // --- Level 4: Atomic global update ---
      //
      // Multiple blocks process the same rows (each block covers a different column tile).
      // The global output min_output[row] must accumulate the minimum across all blocks.
      // Two compile-time paths handle this based on the size of OutT:

      if constexpr (sizeof(OutT) == sizeof(unsigned long long)) {
        // Path A: Lock-free 64-bit CAS (for 8-byte OutT, e.g. KeyValuePair<int32, float>)
        //
        // All lanes work in parallel -- no serialization needed. Each lane:
        //   1. Reads the current global KVPair as a raw 64-bit word
        //   2. If our candidate has a smaller distance, pack it and atomicCAS
        //   3. On CAS failure (another block updated concurrently), re-read and retry
        //   4. Stop when CAS succeeds or the global value is already smaller
        //
        // This is fully lock-free: no mutex, no deadlock risk, both key and value
        // are written as a single atomic 64-bit word.
        for (ck_tile::index_t idx = 0; idx < num_lane_rows; ++idx) {
          ck_tile::index_t const global_row = i_m + lane_rows[idx];
          if (global_row < M) {
            KVPair candidate = lane_final_min[idx];
            unsigned long long* addr =
              reinterpret_cast<unsigned long long*>(min_output + global_row);
            unsigned long long old_val = *addr;
            KVPair current;
            __builtin_memcpy(&current, &old_val, sizeof(KVPair));

            while (candidate.value < current.value) {
              unsigned long long new_val;
              __builtin_memcpy(&new_val, &candidate, sizeof(unsigned long long));
              unsigned long long prev = atomicCAS(addr, old_val, new_val);
              if (prev == old_val) break;
              old_val = prev;
              __builtin_memcpy(&current, &prev, sizeof(KVPair));
            }
          }
        }
      } else {
        // Path B: Mutex-serialized update (for >8-byte OutT, e.g. KeyValuePair<int64, float>)
        //
        // The KVPair exceeds 64 bits and cannot be atomically updated with a single CAS.
        // A per-row spinlock mutex is used instead. Lanes are serialized within the warp
        // to prevent deadlock from AMD wavefront lockstep execution:
        //
        //   Without serialization, lane A could spin on mutex[X] (held by block B)
        //   while lane B (executing in lockstep) holds mutex[Y] that block B needs,
        //   creating a circular dependency. By activating only one lane at a time,
        //   no lane can be stalled while holding a mutex.
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
