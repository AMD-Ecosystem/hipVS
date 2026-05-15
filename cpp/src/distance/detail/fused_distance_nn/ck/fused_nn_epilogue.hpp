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
// This operates directly on the MFMA accumulator registers using CK's
// sweep_tile_span + get_x_indices_from_distributed_indices APIs, eliminating
// the LDS layout shuffle entirely.
//
// Key components:
// - Fused distance + intra-thread reduction: sweep accumulator, compute distance, argmin
// - Intra-wavefront XOR reduction: shuffle across N-lanes sharing the same row
// - Cross-wavefront LDS reduction: collect per-wavefront row minima
// - Atomic update: mutex-protected or CAS update to global output

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <raft/core/kvp.hpp>

#include <cstddef>
#include <type_traits>

namespace cuvs::distance::detail {

// Problem config matching the pairwise epilogue structure.
template <typename IdxType_,
          ck_tile::index_t kM_,
          ck_tile::index_t kN_,
          ck_tile::index_t MWave_,
          ck_tile::index_t NWave_,
          ck_tile::index_t MPerXdl_,
          ck_tile::index_t NPerXdl_,
          ck_tile::index_t KPerXdl_>
struct FusedDistanceNNCkEpilogueProblem {
  using AccDataType = float;
  using ODataType   = float;
  using IdxType     = ck_tile::remove_cvref_t<IdxType_>;

  static constexpr ck_tile::index_t kBlockSize = MWave_ * NWave_ * ck_tile::get_warp_size();
  static constexpr ck_tile::index_t kMPerBlock = kM_;
  static constexpr ck_tile::index_t kNPerBlock = kN_;
  static constexpr ck_tile::index_t MWave      = MWave_;
  static constexpr ck_tile::index_t NWave      = NWave_;
  static constexpr ck_tile::index_t MPerXdl    = MPerXdl_;
  static constexpr ck_tile::index_t NPerXdl    = NPerXdl_;
  static constexpr ck_tile::index_t KPerXdl    = KPerXdl_;

  static constexpr ck_tile::index_t MPerIteration = MPerXdl * MWave;
  static constexpr ck_tile::index_t NPerIteration = NPerXdl * NWave;
  static constexpr ck_tile::index_t MRepeat       = kMPerBlock / (MPerXdl * MWave);
  static constexpr ck_tile::index_t NRepeat       = kNPerBlock / (NPerXdl * NWave);

  using ELayout = ck_tile::tensor_layout::gemm::RowMajor;
};

// Fused Distance NN Epilogue: distances + row-wise reduction + atomic update
template <typename Problem_, typename ElementWiseOp_, typename ReduceOpT_, typename KVPReduceOpT_>
struct FusedDistanceNNCkEpilogue {
  using Problem       = Problem_;
  using ElementWiseOp = ElementWiseOp_;
  using ReduceOpT     = ReduceOpT_;
  using KVPReduceOpT  = KVPReduceOpT_;
  using AccDataType   = typename Problem::AccDataType;
  using ODataType     = typename Problem::ODataType;
  using IdxType       = typename Problem::IdxType;
  using KVPair        = raft::KeyValuePair<IdxType, ODataType>;

  // Guard against silent truncation of the KVPair key: the global column
  // index is computed in ck_tile::index_t (i32), and a narrower IdxType
  // would wrap negative without warning.
  static_assert(sizeof(IdxType) >= sizeof(ck_tile::index_t),
                "IdxType must be >= ck_tile::index_t; otherwise the KVPair key "
                "silently narrows when storing the global column index.");

  static constexpr ck_tile::index_t kBlockSize    = Problem::kBlockSize;
  static constexpr ck_tile::index_t kMPerBlock    = Problem::kMPerBlock;
  static constexpr ck_tile::index_t kNPerBlock    = Problem::kNPerBlock;
  static constexpr ck_tile::index_t MWave         = Problem::MWave;
  static constexpr ck_tile::index_t NWave         = Problem::NWave;
  static constexpr ck_tile::index_t MPerXdl       = Problem::MPerXdl;
  static constexpr ck_tile::index_t NPerXdl       = Problem::NPerXdl;
  static constexpr ck_tile::index_t MPerIteration = Problem::MPerIteration;
  static constexpr ck_tile::index_t NPerIteration = Problem::NPerIteration;
  static constexpr ck_tile::index_t MRepeat       = Problem::MRepeat;
  static constexpr ck_tile::index_t NRepeat       = Problem::NRepeat;

  // Cross-wavefront reduction LDS: only NWave entries per row (warps sharing same rows)
  __host__ __device__ static constexpr ck_tile::index_t GetSmemSize()
  {
    constexpr ck_tile::index_t reduce_size = kMPerBlock * NWave * sizeof(KVPair);
    return reduce_size;
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
  // Optimized pipeline (no LDS shuffle):
  //   Phase 1+2: Fused distance computation + intra-thread argmin (direct from MFMA registers)
  //   Phase 3:   Intra-wavefront XOR reduction across N-lanes
  //   Phase 4:   Cross-wavefront LDS reduction
  //   Phase 5:   Final atomic global update
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
                             ODataType maxVal,
                             ck_tile::index_t n_offset = 0) const
  {
    // ===========================================================================================
    // Compute kRowsPerThread: the number of unique rows each thread owns in the MFMA accumulator.
    //
    // For MFMA 16x16xK with 64-thread warps:
    //   - 4 rows per lane within one XDL (kCM1PerLane = MPerXdl / 4 = 4)
    //   - MRepeat XDL repeats along M
    //   Result: MRepeat * 4 = 8 rows per thread (for default config: MRepeat=2)
    // ===========================================================================================

    constexpr ck_tile::index_t kCM1PerLane    = MPerXdl / 4;  // 4 for MPerXdl=16
    constexpr ck_tile::index_t kRowsPerThread = MRepeat * kCM1PerLane;

    KVPair thread_row_min[kRowsPerThread];
    ck_tile::index_t thread_block_rows[kRowsPerThread];
    for (ck_tile::index_t r = 0; r < kRowsPerThread; ++r) {
      thread_row_min[r]    = {0, maxVal};
      thread_block_rows[r] = 0;
    }

    // ===========================================================================================
    // Phase 1+2: Fused distance computation + intra-thread argmin
    //
    // Iterate the MFMA accumulator using sweep_tile_span. For each element:
    //   - Get (block_row, block_col) via get_x_indices_from_distributed_indices
    //   - Load norms from global memory (L1-cached scalar loads)
    //   - Compute distance via elfunc_
    //   - Reduce into compact thread_row_min[kRowsPerThread] array
    //
    // norm_x depends only on the row (outer span) — it is loaded once per row
    // and reused across all N-column iterations, avoiding redundant VMEM loads.
    // ===========================================================================================

    constexpr auto acc_spans = OAccTile::get_distributed_spans();
    ck_tile::index_t row_idx = 0;

    using Dim0 = ck_tile::number<0>;
    using Dim1 = ck_tile::number<1>;
    ck_tile::sweep_tile_span(acc_spans[Dim0{}], [&](auto idx0) {
      // norm_x is constant across the inner (column) sweep — load once per row.
      ODataType nx{};
      ck_tile::index_t global_row = M;  // default out-of-bounds
      bool row_loaded             = false;

      ck_tile::sweep_tile_span(acc_spans[Dim1{}], [&](auto idx1) {
        constexpr auto dstr_idx = ck_tile::make_tuple(idx0, idx1);
        const auto tile_coord   = ck_tile::get_x_indices_from_distributed_indices(
          o_acc_tile.get_tile_distribution(), dstr_idx);
        auto block_col = tile_coord.at(ck_tile::number<1>{});

        // On the first column iteration, extract row info and load norm_x.
        // block_row is constant across the inner (column) sweep for MFMA layout.
        if (!row_loaded) {
          auto block_row             = tile_coord.at(ck_tile::number<0>{});
          global_row                 = i_m + block_row;
          thread_block_rows[row_idx] = block_row;
          if (global_row < M) { nx = norm_x_ptr[global_row]; }
          row_loaded = true;
        }

        ck_tile::index_t global_col = i_n + block_col;

        if (global_row < M && global_col < N) {
          AccDataType dot = o_acc_tile[dstr_idx];
          ODataType ny    = norm_y_ptr[global_col];
          ODataType dist;
          elfunc_(dist, static_cast<ODataType>(dot), nx, ny);

          KVPair candidate        = {static_cast<IdxType>(global_col + n_offset), dist};
          thread_row_min[row_idx] = pair_red_op_(global_row, candidate, thread_row_min[row_idx]);
        }
      });
      row_idx++;
    });

    // ===========================================================================================
    // Phase 3: Intra-wavefront XOR reduction
    //
    // Reduce across the N-lanes (threads sharing the same row within a wavefront).
    // For MFMA 16x16, lane_id = m_lane*16 + n_lane, so XOR offsets {1,2,4,8}
    // stay within the 16-lane N-group.
    // ===========================================================================================

    for (ck_tile::index_t r = 0; r < kRowsPerThread; ++r) {
      ODataType val = thread_row_min[r].value;
      IdxType key   = thread_row_min[r].key;

      for (ck_tile::index_t offset = 1; offset < NPerXdl; offset <<= 1) {
        ODataType other_val = __shfl_xor(val, offset, ck_tile::get_warp_size());
        IdxType other_key   = __shfl_xor(key, offset, ck_tile::get_warp_size());
        KVPair other        = {other_key, other_val};
        KVPair current      = {key, val};
        KVPair result       = pair_red_op_(0, other, current);
        val                 = result.value;
        key                 = result.key;
      }

      thread_row_min[r] = {key, val};
    }

    // ===========================================================================================
    // Phase 4: Cross-wavefront LDS reduction
    //
    // Only warps along the N dimension share rows. Index LDS by n_warp (not warp_id)
    // so that exactly NWave entries exist per row, all of which are written.
    // ===========================================================================================

    auto reduce_smem = reinterpret_cast<KVPair*>(p_smem);

    ck_tile::index_t const lane_id = threadIdx.x % ck_tile::get_warp_size();
    ck_tile::index_t const warp_id = threadIdx.x / ck_tile::get_warp_size();
    ck_tile::index_t const n_lane  = lane_id % NPerXdl;
    ck_tile::index_t const n_warp  = warp_id % NWave;

    // Lane 0 of N-lane group writes per-row results to LDS
    // LDS layout: [kMPerBlock x NWave], indexed by [block_row * NWave + n_warp]
    if (n_lane == 0) {
      for (ck_tile::index_t r = 0; r < kRowsPerThread; ++r) {
        ck_tile::index_t block_row              = thread_block_rows[r];
        reduce_smem[block_row * NWave + n_warp] = thread_row_min[r];
      }
    }

    ck_tile::block_sync_lds();

    // ===========================================================================================
    // Phase 5: Final reduction + atomic global update
    //
    // wavefront 0 reduces across NWave entries per row and atomically updates global output.
    // ===========================================================================================

    if (warp_id == 0) {
      for (ck_tile::index_t r = lane_id; r < kMPerBlock; r += ck_tile::get_warp_size()) {
        ck_tile::index_t const global_row = i_m + r;
        if (global_row >= M) continue;

        // Reduce across NWave warps that share this row
        KVPair final_min = reduce_smem[r * NWave];
        for (ck_tile::index_t w = 1; w < NWave; ++w) {
          final_min = pair_red_op_(global_row, reduce_smem[r * NWave + w], final_min);
        }

        // Skip rows that were never touched (still at maxVal)
        if (final_min.value >= maxVal) continue;

        // Gate the lock-free CAS path by KVPair shape (not just size) so that
        // OutT=double or other 8-byte types that aren't a KVPair fall through to
        // the mutex path. The CAS path hardcodes value-only argmin semantics and
        // does NOT call red_op_; it is only correct when red_op_ performs a simple
        // "keep the smaller .value" reduction.
        constexpr bool kCanUseLockFreeCas = std::is_same_v<OutT, KVPair> &&
                                            sizeof(KVPair) == sizeof(unsigned long long) &&
                                            alignof(KVPair) >= alignof(unsigned long long);

        static_assert(!kCanUseLockFreeCas || offsetof(KVPair, value) == sizeof(IdxType),
                      "CAS path assumes KVPair = {key, value} layout; update if changed.");

        if constexpr (kCanUseLockFreeCas) {
          // Lock-free 64-bit CAS path: all lanes work in parallel
          unsigned long long* addr = reinterpret_cast<unsigned long long*>(min_output + global_row);
          unsigned long long old_val = *addr;
          KVPair current;
          __builtin_memcpy(&current, &old_val, sizeof(KVPair));

          while (final_min.value < current.value) {
            unsigned long long new_val;
            __builtin_memcpy(&new_val, &final_min, sizeof(unsigned long long));
            unsigned long long prev = atomicCAS(addr, old_val, new_val);
            if (prev == old_val) break;
            old_val = prev;
            __builtin_memcpy(&current, &prev, sizeof(KVPair));
          }
        } else {
          // Mutex path: lanes must be serialized within the wavefront to prevent
          // deadlock from AMD wavefront lockstep execution. Without serialization,
          // a lane that already acquired its mutex would be forced to re-execute
          // the CAS (due to another lane spinning), see value 1 (that it set),
          // and spin forever thinking someone else holds the lock.
          for (ck_tile::index_t j = 0; j < ck_tile::get_warp_size(); ++j) {
            if (lane_id == j) {
              while (atomicCAS(mutex + global_row, 0, 1) == 1)
                ;
              __threadfence();
              red_op_(global_row, min_output + global_row, final_min);
              __threadfence();
              atomicCAS(mutex + global_row, 1, 0);
            }
          }
        }
      }
    }
  }
};

}  // namespace cuvs::distance::detail
