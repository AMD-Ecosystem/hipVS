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

// Fused Distance NN CK kernel: GEMM + distance epilogue with row-wise reduction.
// This follows the same pattern as pairwise_matrix/ck/ck_kernel.hpp but outputs
// the minimum (index, distance) pair per row instead of the full distance matrix.

#pragma once

#include "fused_nn_epilogue.hpp"

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <limits>
#include <tuple>

namespace cuvs::distance::detail {

// Tile configuration for fused distance NN GEMM (default for K >= 32)
struct FusedDistanceNNGemmConfig {
  static constexpr ck_tile::index_t M_Tile      = 64;
  static constexpr ck_tile::index_t N_Tile      = 64;
  static constexpr ck_tile::index_t K_Tile      = 32;
  static constexpr ck_tile::index_t M_Warp      = 2;
  static constexpr ck_tile::index_t N_Warp      = 2;
  static constexpr ck_tile::index_t K_Warp      = 1;
  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 16;
  static constexpr bool DoubleSmemBuffer        = false;
  static constexpr auto Scheduler               = ck_tile::GemmPipelineScheduler::Intrawave;
};

// Optimized configuration for fp32 with K < 32, uses MFMA 16x16x4 (minimum K_Warp_Tile=4)
struct FusedDistanceNNGemmConfigSkinnyF32 {
  static constexpr ck_tile::index_t M_Tile      = 64;
  static constexpr ck_tile::index_t N_Tile      = 64;
  static constexpr ck_tile::index_t K_Tile      = 8;  // 4x smaller than default 32
  static constexpr ck_tile::index_t M_Warp      = 2;
  static constexpr ck_tile::index_t N_Warp      = 2;
  static constexpr ck_tile::index_t K_Warp      = 1;
  static constexpr ck_tile::index_t M_Warp_Tile = 16;
  static constexpr ck_tile::index_t N_Warp_Tile = 16;
  static constexpr ck_tile::index_t K_Warp_Tile = 4;  // Minimum for fp32 MFMA 16x16x4
  static constexpr bool DoubleSmemBuffer        = false;
  static constexpr auto Scheduler               = ck_tile::GemmPipelineScheduler::Intrawave;
};

// Template alias for GEMM shape parameterized by config
template <typename Config>
using FusedNNGemmShapeT = ck_tile::TileGemmShape<
  ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
  ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
  ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;

// Backward-compatible alias using default config
using FusedNNGemmShape = FusedNNGemmShapeT<FusedDistanceNNGemmConfig>;

static constexpr ck_tile::index_t FusedNNTilePartitionerGroupNum = 8;
static constexpr ck_tile::index_t FusedNNTilePartitionerM01      = 4;

// Template alias for tile partitioner parameterized by config
template <typename Config>
using FusedNNTilePartitionerT =
  ck_tile::GemmSpatiallyLocalTilePartitioner<FusedNNGemmShapeT<Config>,
                                             FusedNNTilePartitionerGroupNum,
                                             FusedNNTilePartitionerM01>;

// Backward-compatible alias using default config
using FusedNNTilePartitioner = FusedNNTilePartitionerT<FusedDistanceNNGemmConfig>;

template <typename DataT>
static constexpr bool isSupportedFusedNN()
{
  return std::is_same_v<DataT, float>;  // Only float supported, double uses SIMT fallback
}

template <typename InputLayoutT>
struct FusedNNInputLayoutTraits;

template <>
struct FusedNNInputLayoutTraits<ck_tile::tensor_layout::gemm::RowMajor> {
  using A_Layout = ck_tile::tensor_layout::gemm::RowMajor;
  using B_Layout = ck_tile::tensor_layout::gemm::ColumnMajor;
};

template <>
struct FusedNNInputLayoutTraits<ck_tile::tensor_layout::gemm::ColumnMajor> {
  using A_Layout = ck_tile::tensor_layout::gemm::ColumnMajor;
  using B_Layout = ck_tile::tensor_layout::gemm::RowMajor;
};

template <typename DataT,
          typename InputLayout = ck_tile::tensor_layout::gemm::RowMajor,
          typename Config      = FusedDistanceNNGemmConfig>
using FusedNNGemmUniversalTraitsT =
  ck_tile::TileGemmUniversalTraits<true,
                                   true,
                                   true,
                                   Config::DoubleSmemBuffer,
                                   typename FusedNNInputLayoutTraits<InputLayout>::A_Layout,
                                   typename FusedNNInputLayoutTraits<InputLayout>::B_Layout,
                                   ck_tile::tensor_layout::gemm::RowMajor>;

// Backward-compatible alias using default config
template <typename DataT, typename InputLayout = ck_tile::tensor_layout::gemm::RowMajor>
using FusedNNGemmUniversalTraits =
  FusedNNGemmUniversalTraitsT<DataT, InputLayout, FusedDistanceNNGemmConfig>;

template <typename DataT,
          typename DistT,
          typename InputLayout = ck_tile::tensor_layout::gemm::RowMajor,
          bool ScalarVec       = false,
          typename Config      = FusedDistanceNNGemmConfig>
using FusedNNUniversalGemmProblemT =
  ck_tile::UniversalGemmPipelineProblem<DataT,
                                        DataT,
                                        DistT,
                                        FusedNNGemmShapeT<Config>,
                                        FusedNNGemmUniversalTraitsT<DataT, InputLayout, Config>,
                                        Config::Scheduler,
                                        true,
                                        ck_tile::TailNumber::Full,
                                        ck_tile::element_wise::PassThrough,
                                        ck_tile::element_wise::PassThrough,
                                        DataT,
                                        ScalarVec,
                                        1,
                                        1>;

// Backward-compatible alias using default config
template <typename DataT,
          typename DistT,
          typename InputLayout = ck_tile::tensor_layout::gemm::RowMajor,
          bool ScalarVec       = false>
using FusedNNUniversalGemmProblem =
  FusedNNUniversalGemmProblemT<DataT, DistT, InputLayout, ScalarVec, FusedDistanceNNGemmConfig>;

template <typename DataT,
          typename DistT,
          typename InputLayout,
          bool ScalarVec  = false,
          typename Config = FusedDistanceNNGemmConfig>
using FusedNNGemmPipelineT = ck_tile::GemmPipelineAgBgCrCompV3<
  FusedNNUniversalGemmProblemT<DataT, DistT, InputLayout, ScalarVec, Config>>;

// Backward-compatible alias using default config
template <typename DataT, typename DistT, typename InputLayout, bool ScalarVec = false>
using FusedNNGemmPipeline =
  FusedNNGemmPipelineT<DataT, DistT, InputLayout, ScalarVec, FusedDistanceNNGemmConfig>;

// Dummy epilogue for CK's tensor view creation (we don't output to D tensors)
template <typename DataT, typename DistT, bool ScalarVec>
struct FusedDistanceNNCkDummyEpilogue {
  using ODataType  = DistT;
  using DsDataType = ck_tile::tuple<>;
  using DsLayout   = ck_tile::tuple<>;
  static constexpr ck_tile::memory_operation_enum MemoryOperation =
    ck_tile::memory_operation_enum::set;

  __host__ __device__ static constexpr ck_tile::index_t GetVectorSizeC()
  {
    return ScalarVec ? 1 : std::min(4, static_cast<int>(16 / sizeof(ODataType)));
  }
  template <ck_tile::index_t I>
  __host__ __device__ static constexpr ck_tile::index_t GetVectorSizeD(ck_tile::number<I>)
  {
    return 1;
  }
  __host__ __device__ static constexpr ck_tile::index_t GetSmemSize() { return 0; }
};

// Kernel arguments for fused distance NN
template <typename DataT, typename DistT, typename IdxT, typename OutT>
struct FusedDistanceNNCkKernelArgs {
  DataT const* a_ptr;
  DataT const* b_ptr;
  OutT* min_ptr;
  DistT const* norm_x_ptr;
  DistT const* norm_y_ptr;
  int* mutex_ptr;
  ck_tile::index_t M;
  ck_tile::index_t N;
  ck_tile::index_t K;
  ck_tile::index_t stride_A;
  ck_tile::index_t stride_B;
  DistT maxVal;
};

// Template epilogue problem type alias parameterized by config
template <typename DataT, typename DistT, typename IdxT, bool ScalarVec, typename Config>
using FusedNNEpilogueProblemForT =
  FusedDistanceNNCkEpilogueProblem<float,
                                   DistT,
                                   IdxT,
                                   FusedNNGemmShapeT<Config>::kM,
                                   FusedNNGemmShapeT<Config>::kN,
                                   FusedNNGemmShapeT<Config>::BlockWarps::at(ck_tile::number<0>{}),
                                   FusedNNGemmShapeT<Config>::BlockWarps::at(ck_tile::number<1>{}),
                                   FusedNNGemmShapeT<Config>::WarpTile::at(ck_tile::number<0>{}),
                                   FusedNNGemmShapeT<Config>::WarpTile::at(ck_tile::number<1>{}),
                                   FusedNNGemmShapeT<Config>::WarpTile::at(ck_tile::number<2>{}),
                                   ScalarVec,
                                   1,
                                   1>;

// Backward-compatible alias using default config
template <typename DataT, typename DistT, typename IdxT, bool ScalarVec>
using FusedNNEpilogueProblemFor =
  FusedNNEpilogueProblemForT<DataT, DistT, IdxT, ScalarVec, FusedDistanceNNGemmConfig>;

// Main fused distance NN kernel
template <typename DataT,
          typename DistT,
          typename IdxT,
          typename OutT,
          typename Layout,
          bool ScalarVec,
          typename ElementWiseOp,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename Config = FusedDistanceNNGemmConfig>  // Default config for compatibility
struct FusedDistanceNNCkKernel {
  using GemmPipeline  = FusedNNGemmPipelineT<DataT, DistT, Layout, ScalarVec, Config>;
  using DummyEpilogue = FusedDistanceNNCkDummyEpilogue<DataT, DistT, ScalarVec>;
  using HelperKernel =
    ck_tile::UniversalGemmKernel<FusedNNTilePartitionerT<Config>, GemmPipeline, DummyEpilogue>;

  using ADataType  = DataT;
  using BDataType  = DataT;
  using EDataType  = DistT;
  using KernelArgs = FusedDistanceNNCkKernelArgs<DataT, DistT, IdxT, OutT>;

  using ALayout = typename FusedNNInputLayoutTraits<Layout>::A_Layout;
  using BLayout = typename FusedNNInputLayoutTraits<Layout>::B_Layout;

  static constexpr ck_tile::index_t MPerBlock = FusedNNTilePartitionerT<Config>::MPerBlock;
  static constexpr ck_tile::index_t NPerBlock = FusedNNTilePartitionerT<Config>::NPerBlock;
  static constexpr ck_tile::index_t KPerBlock = FusedNNTilePartitionerT<Config>::KPerBlock;

  using EpilogueType =
    FusedDistanceNNCkEpilogue<FusedNNEpilogueProblemForT<DataT, DistT, IdxT, ScalarVec, Config>,
                              ElementWiseOp,
                              ReduceOpT,
                              KVPReduceOpT>;

  static constexpr ck_tile::index_t kBlockSize = HelperKernel::kBlockSize;

  __host__ static constexpr auto GridSize(ck_tile::index_t M,
                                          ck_tile::index_t N,
                                          ck_tile::index_t KBatch) -> dim3
  {
    return HelperKernel::GridSize(M, N, KBatch);
  }

  __host__ static constexpr auto BlockSize() -> dim3 { return HelperKernel::BlockSize(); }

  __host__ static constexpr ck_tile::index_t GetSmemSize()
  {
    return ck_tile::max(GemmPipeline::GetSmemSize(), EpilogueType::GetSmemSize());
  }

  __host__ static auto IsSupportedArgument(KernelArgs const& kargs) -> bool
  {
    auto const universal_args = to_universal_args(kargs);
    return HelperKernel::IsSupportedArgument(universal_args);
  }

 private:
  __host__ __device__ static constexpr ck_tile::UniversalGemmKernelArgs<1, 1, 0> to_universal_args(
    KernelArgs const& k)
  {
    return ck_tile::UniversalGemmKernelArgs<1, 1, 0>{
      {reinterpret_cast<DataT const*>(k.a_ptr)},
      {reinterpret_cast<DataT const*>(k.b_ptr)},
      {},
      nullptr,  // No output tensor E for fused NN
      k.M,
      k.N,
      k.K,
      {k.stride_A},
      {k.stride_B},
      {},
      0,  // stride_E unused
      1,  // k_batch
    };
  }

 public:
  ElementWiseOp elfunc_;
  ReduceOpT red_op_;
  KVPReduceOpT pair_red_op_;

  __host__ __device__ FusedDistanceNNCkKernel(ElementWiseOp elfunc     = {},
                                              ReduceOpT red_op         = {},
                                              KVPReduceOpT pair_red_op = {})
    : elfunc_(elfunc), red_op_(red_op), pair_red_op_(pair_red_op)
  {
  }

  __device__ void operator()(KernelArgs kargs) const
  {
    auto const [i_m, i_n] = get_tile_indices(kargs);

    __shared__ char smem_ptr_0[GetSmemSize()];

    // Create tensor views with scalar loads (vector size 1) to support arbitrary K
    auto const a_tensor_view = make_a_tensor_view(kargs.a_ptr, kargs.M, kargs.K, kargs.stride_A);
    auto const b_tensor_view = make_b_tensor_view(kargs.b_ptr, kargs.N, kargs.K, kargs.stride_B);

    // Pad tensor views to handle out-of-bounds K accesses
    auto const a_pad_view = make_a_pad_view(a_tensor_view);
    auto const b_pad_view = make_b_pad_view(b_tensor_view);

    // Create tile windows
    auto const a_block_window = make_a_block_window(a_pad_view, i_m);
    auto const b_block_window = make_b_block_window(b_pad_view, i_n);

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
             kargs.maxVal);
  }

 private:
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

  // Create A tensor view with scalar loads (vector size 1) for arbitrary K support
  __device__ static auto make_a_tensor_view(ADataType const* a_ptr,
                                            ck_tile::index_t M,
                                            ck_tile::index_t K,
                                            ck_tile::index_t stride_A)
  {
    if constexpr (std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>) {
      return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        a_ptr,
        ck_tile::make_tuple(M, K),
        ck_tile::make_tuple(stride_A, 1),
        ck_tile::number<1>{},  // Vector size 1 for scalar loads
        ck_tile::number<1>{});
    } else {
      return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        a_ptr,
        ck_tile::make_tuple(K, M),
        ck_tile::make_tuple(stride_A, 1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});
    }
  }

  // Create B tensor view with scalar loads (vector size 1) for arbitrary K support
  __device__ static auto make_b_tensor_view(BDataType const* b_ptr,
                                            ck_tile::index_t N,
                                            ck_tile::index_t K,
                                            ck_tile::index_t stride_B)
  {
    if constexpr (std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>) {
      return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        b_ptr,
        ck_tile::make_tuple(N, K),
        ck_tile::make_tuple(stride_B, 1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});
    } else {
      return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        b_ptr,
        ck_tile::make_tuple(K, N),
        ck_tile::make_tuple(stride_B, 1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});
    }
  }

  // Pad A tensor view to handle out-of-bounds K accesses
  template <typename TensorView>
  __device__ static auto make_a_pad_view(TensorView const& a_view)
  {
    if constexpr (std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>) {
      return ck_tile::pad_tensor_view(
        a_view,
        ck_tile::make_tuple(ck_tile::number<MPerBlock>{}, ck_tile::number<KPerBlock>{}),
        ck_tile::sequence<true, true>{});  // Pad both M and K
    } else {
      return ck_tile::pad_tensor_view(
        a_view,
        ck_tile::make_tuple(ck_tile::number<KPerBlock>{}, ck_tile::number<MPerBlock>{}),
        ck_tile::sequence<true, true>{});
    }
  }

  // Pad B tensor view to handle out-of-bounds K accesses
  template <typename TensorView>
  __device__ static auto make_b_pad_view(TensorView const& b_view)
  {
    if constexpr (std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>) {
      return ck_tile::pad_tensor_view(
        b_view,
        ck_tile::make_tuple(ck_tile::number<NPerBlock>{}, ck_tile::number<KPerBlock>{}),
        ck_tile::sequence<true, true>{});
    } else {
      return ck_tile::pad_tensor_view(
        b_view,
        ck_tile::make_tuple(ck_tile::number<KPerBlock>{}, ck_tile::number<NPerBlock>{}),
        ck_tile::sequence<true, true>{});
    }
  }

  // Create A block window
  template <typename PadView>
  __device__ static auto make_a_block_window(PadView const& a_pad_view, ck_tile::index_t i_m)
  {
    if constexpr (std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>) {
      return ck_tile::make_tile_window(
        a_pad_view,
        ck_tile::make_tuple(ck_tile::number<MPerBlock>{}, ck_tile::number<KPerBlock>{}),
        {i_m, 0});
    } else {
      return ck_tile::make_tile_window(
        a_pad_view,
        ck_tile::make_tuple(ck_tile::number<KPerBlock>{}, ck_tile::number<MPerBlock>{}),
        {0, i_m});
    }
  }

  // Create B block window
  template <typename PadView>
  __device__ static auto make_b_block_window(PadView const& b_pad_view, ck_tile::index_t i_n)
  {
    if constexpr (std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>) {
      return ck_tile::make_tile_window(
        b_pad_view,
        ck_tile::make_tuple(ck_tile::number<NPerBlock>{}, ck_tile::number<KPerBlock>{}),
        {i_n, 0});
    } else {
      return ck_tile::make_tile_window(
        b_pad_view,
        ck_tile::make_tuple(ck_tile::number<KPerBlock>{}, ck_tile::number<NPerBlock>{}),
        {0, i_n});
    }
  }
};

}  // namespace cuvs::distance::detail
