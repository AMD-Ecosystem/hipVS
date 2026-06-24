// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// Fused Distance NN CK kernel: GEMM + distance epilogue with row-wise reduction.
// This follows the same pattern as pairwise_matrix/ck/ck_kernel.hpp but outputs
// the minimum (index, distance) pair per row instead of the full distance matrix.

#pragma once

#include "fused_nn_epilogue.hpp"

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <tuple>

namespace cuvs::distance::detail {

// MFMA hardware instruction tags (pure compile-time descriptors of CDNA MFMA shapes).
// Used to make the GemmConfig declare which hardware instruction it intends to use.
namespace mfma {
// clang-format off
struct f32_f32_f32_16x16x16 { static constexpr ck_tile::index_t M = 16, N = 16, K = 16; };
struct f32_f32_f32_16x16x4  { static constexpr ck_tile::index_t M = 16, N = 16, K = 4;  };
// clang-format on
}  // namespace mfma

struct FusedDistanceNNGemmBaseConfig {
  static constexpr ck_tile::index_t M_Tile = 64;
  static constexpr ck_tile::index_t N_Tile = 64;
  static constexpr ck_tile::index_t M_Warp = 2;
  static constexpr ck_tile::index_t N_Warp = 2;
  static constexpr ck_tile::index_t K_Warp = 1;
  static constexpr bool DoubleSmemBuffer   = false;
  static constexpr auto Scheduler          = ck_tile::GemmPipelineScheduler::Intrawave;
};

// Tile configuration for fused distance NN GEMM (default for K >= 32)
struct FusedDistanceNNGemmConfig : public FusedDistanceNNGemmBaseConfig {
  static constexpr ck_tile::index_t K_Tile      = 32;
  using Mfma                                    = mfma::f32_f32_f32_16x16x16;
  static constexpr ck_tile::index_t M_Warp_Tile = Mfma::M;
  static constexpr ck_tile::index_t N_Warp_Tile = Mfma::N;
  static constexpr ck_tile::index_t K_Warp_Tile = Mfma::K;
};

// Optimized configuration for fp32 with 8 < K < 32, uses MFMA 16x16x4 (minimum K_Warp_Tile=4)
struct FusedDistanceNNGemmConfigSkinnyF32 : public FusedDistanceNNGemmBaseConfig {
  using Mfma                                    = mfma::f32_f32_f32_16x16x4;
  static constexpr ck_tile::index_t K_Tile      = 2 * Mfma::K;  // 8
  static constexpr ck_tile::index_t M_Warp_Tile = Mfma::M;
  static constexpr ck_tile::index_t N_Warp_Tile = Mfma::N;
  static constexpr ck_tile::index_t K_Warp_Tile = Mfma::K;
};

// Smallest-tile configuration for fp32 with K <= 8, minimizes wasted work
struct FusedDistanceNNGemmConfigVerySkinnyF32 : public FusedDistanceNNGemmBaseConfig {
  using Mfma                                    = mfma::f32_f32_f32_16x16x4;
  static constexpr ck_tile::index_t K_Tile      = Mfma::K;  // 4
  static constexpr ck_tile::index_t M_Warp_Tile = Mfma::M;
  static constexpr ck_tile::index_t N_Warp_Tile = Mfma::N;
  static constexpr ck_tile::index_t K_Warp_Tile = Mfma::K;
};

// Template alias for GEMM shape parameterized by config
template <typename Config>
using FusedNNGemmShapeT = ck_tile::TileGemmShape<
  ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
  ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
  ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;

// These values are the defaults used in CK, specifically tuned for the MI300X.
// TODO: Revisit, might be worth having this be architecture specific.
static constexpr ck_tile::index_t FusedNNTilePartitionerGroupNum = 8;
static constexpr ck_tile::index_t FusedNNTilePartitionerM01      = 4;

// Template alias for tile partitioner parameterized by config
template <typename Config>
using FusedNNTilePartitionerT =
  ck_tile::GemmSpatiallyLocalTilePartitioner<FusedNNGemmShapeT<Config>,
                                             FusedNNTilePartitionerGroupNum,
                                             FusedNNTilePartitionerM01>;

template <typename DataT>
static constexpr bool isSupportedFusedNN()
{
  return std::is_same_v<DataT, float>;  // Only float supported, double uses SIMT fallback
}

template <typename Config = FusedDistanceNNGemmConfig>
using FusedNNGemmUniversalTraitsT =
  ck_tile::TileGemmUniversalTraits<true,
                                   true,
                                   true,
                                   Config::DoubleSmemBuffer,
                                   ck_tile::tensor_layout::gemm::RowMajor,
                                   ck_tile::tensor_layout::gemm::ColumnMajor,
                                   ck_tile::tensor_layout::gemm::RowMajor>;

// Only float is supported (guarded by isSupportedFusedNN), ScalarVec is always false.
template <typename Config = FusedDistanceNNGemmConfig>
using FusedNNUniversalGemmProblemT =
  ck_tile::UniversalGemmPipelineProblem<float,
                                        float,
                                        float,
                                        FusedNNGemmShapeT<Config>,
                                        FusedNNGemmUniversalTraitsT<Config>,
                                        Config::Scheduler,
                                        true,
                                        ck_tile::TailNumber::Full,
                                        ck_tile::element_wise::PassThrough,
                                        ck_tile::element_wise::PassThrough,
                                        float,
                                        false,
                                        1,
                                        1>;

template <typename Config = FusedDistanceNNGemmConfig>
using FusedNNGemmPipelineT =
  ck_tile::GemmPipelineAgBgCrCompV3<FusedNNUniversalGemmProblemT<Config>>;

// Kernel arguments for fused distance NN. DataT/DistT are always float.
template <typename IdxT, typename OutT>
struct FusedDistanceNNCkKernelArgs {
  float const* a_ptr;
  float const* b_ptr;
  OutT* min_ptr;
  float const* norm_x_ptr;
  float const* norm_y_ptr;
  int* mutex_ptr;
  ck_tile::index_t M;
  ck_tile::index_t N;
  ck_tile::index_t K;
  ck_tile::index_t stride_A;
  ck_tile::index_t stride_B;
  ck_tile::index_t n_offset;  // Global column offset for N-batching
};

// Template epilogue problem type alias parameterized by config.
// Only float data type is supported; AccDataType is hardcoded to float in the problem.
template <typename IdxT, typename Config>
using FusedNNEpilogueProblemForT =
  FusedDistanceNNCkEpilogueProblem<IdxT,
                                   FusedNNGemmShapeT<Config>::kM,
                                   FusedNNGemmShapeT<Config>::kN,
                                   FusedNNGemmShapeT<Config>::BlockWarps::at(ck_tile::number<0>{}),
                                   FusedNNGemmShapeT<Config>::BlockWarps::at(ck_tile::number<1>{}),
                                   FusedNNGemmShapeT<Config>::WarpTile::at(ck_tile::number<0>{}),
                                   FusedNNGemmShapeT<Config>::WarpTile::at(ck_tile::number<1>{}),
                                   FusedNNGemmShapeT<Config>::WarpTile::at(ck_tile::number<2>{})>;

// Main fused distance NN kernel.
// Only float data type is supported; ScalarVec is always false.
template <typename IdxT,
          typename OutT,
          typename ElementWiseOp,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename Config = FusedDistanceNNGemmConfig>
class FusedDistanceNNComposableKernel {
 public:
  using GemmPipeline = FusedNNGemmPipelineT<Config>;
  using KernelArgs   = FusedDistanceNNCkKernelArgs<IdxT, OutT>;

  static constexpr auto kBlockSize = GemmPipeline::BlockSize;

  __host__ __device__ FusedDistanceNNComposableKernel(ElementWiseOp elfunc     = {},
                                                      ReduceOpT red_op         = {},
                                                      KVPReduceOpT pair_red_op = {})
    : elfunc_(elfunc), red_op_(red_op), pair_red_op_(pair_red_op)
  {
  }
  __device__ void operator()(KernelArgs kargs) const
  {
    auto const [i_m, i_n] = get_tile_indices(kargs);

    __shared__ alignas(raft::KeyValuePair<IdxT, float>) hip::std::byte smem_ptr_0[GetSmemSize()];

    // Helpers to build CK tensor view → pad view → tile window
    auto make_tensor_view = [&](auto const* ptr, auto rows, auto cols, auto stride) {
      return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        ptr,
        ck_tile::make_tuple(rows, cols),
        ck_tile::make_tuple(stride, 1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});
    };

    auto make_pad_view = [](auto const& view, auto row_tile, auto col_tile) {
      return ck_tile::pad_tensor_view(
        view, ck_tile::make_tuple(row_tile, col_tile), ck_tile::sequence<true, true>{});
    };

    auto make_block_window = [](auto const& pad_view, auto row_tile, auto col_tile, auto offset) {
      return ck_tile::make_tile_window(pad_view, ck_tile::make_tuple(row_tile, col_tile), offset);
    };

    auto const a_block_window = make_block_window(
      make_pad_view(make_tensor_view(kargs.a_ptr, kargs.M, kargs.K, kargs.stride_A),
                    ck_tile::number<GemmPipeline::MPerBlock>{},
                    ck_tile::number<GemmPipeline::KPerBlock>{}),
      ck_tile::number<GemmPipeline::MPerBlock>{},
      ck_tile::number<GemmPipeline::KPerBlock>{},
      ck_tile::multi_index<2>{i_m, 0});

    auto const b_block_window = make_block_window(
      make_pad_view(make_tensor_view(kargs.b_ptr, kargs.N, kargs.K, kargs.stride_B),
                    ck_tile::number<GemmPipeline::NPerBlock>{},
                    ck_tile::number<GemmPipeline::KPerBlock>{}),
      ck_tile::number<GemmPipeline::NPerBlock>{},
      ck_tile::number<GemmPipeline::KPerBlock>{},
      ck_tile::multi_index<2>{i_n, 0});

    ck_tile::index_t const num_loop =
      ck_tile::amd_wave_read_first_lane(FusedNNTilePartitionerT<Config>::GetLoopNum(kargs.K));

    // Run GEMM pipeline
    auto const& c_block_tile =
      GemmPipeline{}.template operator()(a_block_window,
                                         typename GemmPipeline::AElementWise{},
                                         b_block_window,
                                         typename GemmPipeline::BElementWise{},
                                         num_loop,
                                         smem_ptr_0);

    // Run fused NN epilogue: distance computation + row reduction + atomic update
    EpilogueType epilogue(elfunc_, red_op_, pair_red_op_);
    epilogue(kargs.min_ptr,
             kargs.mutex_ptr,
             c_block_tile,
             smem_ptr_0,
             kargs.norm_x_ptr,
             kargs.norm_y_ptr,
             i_m,
             i_n,
             kargs.M,
             kargs.N,
             kargs.n_offset);
  }

  // Derived from UniversalGemmKernel::GridSize, which delegates to
  __host__ static constexpr auto GridSize(ck_tile::index_t M, ck_tile::index_t N) -> dim3
  {
    // Each workgroup processes the entire K dimension for its output tile. No inter-workgroup
    // coordination needed.
    return dim3(FusedNNTilePartitionerT<Config>::GridSize(M, N), 1, 1);
  }

  // Derived from UniversalGemmKernel::BlockSize, which halves thread count on
  // wave32 (RDNA) architectures since each warp is 32 threads instead of 64.
  __host__ static auto BlockSize() -> dim3
  {
    return dim3(ck_tile::is_wave32() ? kBlockSize / 2 : kBlockSize);
  }

 private:
  using EpilogueType = FusedDistanceNNCkEpilogue<FusedNNEpilogueProblemForT<IdxT, Config>,
                                                 ElementWiseOp,
                                                 ReduceOpT,
                                                 KVPReduceOpT>;

  __host__ static constexpr ck_tile::index_t GetSmemSize()
  {
    return ck_tile::max(GemmPipeline::GetSmemSize(), EpilogueType::GetSmemSize());
  }

  __device__ static std::tuple<ck_tile::index_t, ck_tile::index_t> get_tile_indices(
    KernelArgs const& kargs)
  {
    auto const blockId = ck_tile::amd_wave_read_first_lane(blockIdx.x);
    auto const [iM, iN] =
      FusedNNTilePartitionerT<Config>{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
    ck_tile::index_t const i_m =
      ck_tile::amd_wave_read_first_lane(iM * FusedNNTilePartitionerT<Config>::MPerBlock);
    ck_tile::index_t const i_n =
      ck_tile::amd_wave_read_first_lane(iN * FusedNNTilePartitionerT<Config>::NPerBlock);
    return {i_m, i_n};
  }

  ElementWiseOp elfunc_;
  ReduceOpT red_op_;
  KVPReduceOpT pair_red_op_;
};

}  // namespace cuvs::distance::detail
