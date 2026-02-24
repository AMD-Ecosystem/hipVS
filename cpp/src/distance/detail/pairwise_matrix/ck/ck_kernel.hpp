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

// Pairwise distance CK kernel: GEMM + epilogue using norm_x/norm_y directly, without
// materializing the m×n D0 tensor. Folds thrust::tabulate computation into kernel.
//
// --- Relationship to CK multiD GEMM example ---
// This follows a similar pattern to the CK multiD GEMM example:
//   https://github.com/ROCm/rocm-libraries/tree/rocm-7.2.0/projects/composablekernel/example/ck_tile/19_gemm_multi_d
// That example uses UniversalGemmKernel with D0, D1, ... tensors passed to a CShuffleEpilogue
// for fused ops (e.g. bias + activation). We use the same kernel structure (TilePartitioner,
// GemmPipeline, UniversalGemmKernel) but have modified the kernel and epilogue for pairwise
// distance: we do not materialize an m×n D0 tensor. Instead we pass norm_x (length m) and
// norm_y (length n) and our custom epilogue loads per-tile norm values to compute distance
// metrics (cosine, L2_expanded, etc.) without the memory cost of a full D0 matrix.
//
// Key modifications vs. 19_gemm_multi_d:
//   - DummyEpilogue with NumDTensor=0: tensor view creation expects no D tensors
//   - PairwiseDistanceCkKernelArgs: norm_x_ptr, norm_y_ptr instead of D tensor pointers
//   - make_gemm_pointers: ds_ptr is empty; norm data flows to epilogue separately
//   - Custom PairwiseDistanceCkEpilogue: loads norm tiles from 1D vectors, applies
//     ElementWiseOp(e, c, norm_x, norm_y). See epilogue.hpp for CShuffleEpilogue diff.

#pragma once

#include "element_wise_ops.hpp"
#include "epilogue.hpp"

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp>

#include <tuple>

namespace cuvs::distance::detail {

// --- Config: tile/warp/scheduler parameters for distance GEMM ---
template <typename DataT>
static constexpr bool isSupported()
{
  return std::is_same_v<DataT, ck_tile::fp16_t> || std::is_same_v<DataT, float> ||
         std::is_same_v<DataT, ck_tile::bf16_t> || std::is_same_v<DataT, half>;
}

template <typename DataT>
using CkDataType = std::conditional_t<std::is_same_v<DataT, half>, ck_tile::fp16_t, DataT>;

struct GemmDistanceConfig {
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

using GemmShape = ck_tile::TileGemmShape<
  ck_tile::
    sequence<GemmDistanceConfig::M_Tile, GemmDistanceConfig::N_Tile, GemmDistanceConfig::K_Tile>,
  ck_tile::
    sequence<GemmDistanceConfig::M_Warp, GemmDistanceConfig::N_Warp, GemmDistanceConfig::K_Warp>,
  ck_tile::sequence<GemmDistanceConfig::M_Warp_Tile,
                    GemmDistanceConfig::N_Warp_Tile,
                    GemmDistanceConfig::K_Warp_Tile>>;

static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
static constexpr ck_tile::index_t TilePartitionerM01      = 4;

using TilePartitioner = ck_tile::
  GemmSpatiallyLocalTilePartitioner<GemmShape, TilePartitionerGroupNum, TilePartitionerM01>;

template <typename InputLayoutT>
struct InputLayoutTraits;

template <>
struct InputLayoutTraits<ck_tile::tensor_layout::gemm::RowMajor> {
  using A_Layout = ck_tile::tensor_layout::gemm::RowMajor;
  using B_Layout = ck_tile::tensor_layout::gemm::ColumnMajor;
};

template <>
struct InputLayoutTraits<ck_tile::tensor_layout::gemm::ColumnMajor> {
  using A_Layout = ck_tile::tensor_layout::gemm::ColumnMajor;
  using B_Layout = ck_tile::tensor_layout::gemm::RowMajor;
};

template <typename DataT, typename InputLayout = ck_tile::tensor_layout::gemm::RowMajor>
using GemmUniversalTraits =
  ck_tile::TileGemmUniversalTraits<true,
                                   true,
                                   true,
                                   GemmDistanceConfig::DoubleSmemBuffer,
                                   typename InputLayoutTraits<InputLayout>::A_Layout,
                                   typename InputLayoutTraits<InputLayout>::B_Layout,
                                   ck_tile::tensor_layout::gemm::RowMajor>;

template <typename DataT,
          typename DistT,
          typename InputLayout = ck_tile::tensor_layout::gemm::RowMajor,
          bool ScalarVec       = false>
using UniversalGemmProblem =
  ck_tile::UniversalGemmPipelineProblem<DataT,
                                        DataT,
                                        DistT,
                                        GemmShape,
                                        GemmUniversalTraits<DataT, InputLayout>,
                                        GemmDistanceConfig::Scheduler,
                                        true,
                                        ck_tile::TailNumber::Full,
                                        ck_tile::element_wise::PassThrough,
                                        ck_tile::element_wise::PassThrough,
                                        DataT,
                                        ScalarVec,
                                        1,
                                        1>;

template <typename DataT, typename DistT, typename InputLayout, bool ScalarVec = false>
using DistanceGemmPipeline =
  ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem<DataT, DistT, InputLayout, ScalarVec>>;

// --- Kernel and epilogue ---

// [MODIFIED vs 19_gemm_multi_d] DummyEpilogue with NumDTensor=0; no D tensors for view creation
template <typename DataT, typename DistT, bool ScalarVec>
struct PairwiseDistanceCkDummyEpilogue {
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

// [MODIFIED vs 19_gemm_multi_d] norm_x_ptr, norm_y_ptr instead of D tensor pointers
template <typename DataT, typename DistT>
struct PairwiseDistanceCkKernelArgs {
  const DataT* a_ptr;
  const DataT* b_ptr;
  DistT* e_ptr;
  const DistT* norm_x_ptr;
  const DistT* norm_y_ptr;
  ck_tile::index_t M;
  ck_tile::index_t N;
  ck_tile::index_t K;
  ck_tile::index_t stride_A;
  ck_tile::index_t stride_B;
  ck_tile::index_t stride_E;
  ck_tile::index_t k_batch;
};

// Epilogue problem matching CShuffleEpilogue layout for dispatch_ck GemmShape.
template <typename DataT, typename DistT, bool ScalarVec>
using PairwiseDistanceCkEpilogueProblem =
  ck::PairwiseDistanceCkEpilogueProblem<float,
                                        DistT,
                                        GemmShape::kM,
                                        GemmShape::kN,
                                        GemmShape::BlockWarps::at(ck_tile::number<0>{}),
                                        GemmShape::BlockWarps::at(ck_tile::number<1>{}),
                                        GemmShape::WarpTile::at(ck_tile::number<0>{}),
                                        GemmShape::WarpTile::at(ck_tile::number<1>{}),
                                        GemmShape::WarpTile::at(ck_tile::number<2>{}),
                                        ScalarVec,
                                        1,
                                        1>;

// Pairwise distance CK kernel: ElementWiseOp(e, c, norm_x, norm_y).
template <typename DataT, typename DistT, typename Layout, bool ScalarVec, typename ElementWiseOp>
struct PairwiseDistanceCkKernel {
  using GemmPipeline  = DistanceGemmPipeline<DataT, DistT, Layout, ScalarVec>;
  using DummyEpilogue = PairwiseDistanceCkDummyEpilogue<DataT, DistT, ScalarVec>;
  using HelperKernel  = ck_tile::UniversalGemmKernel<TilePartitioner, GemmPipeline, DummyEpilogue>;

  using ADataType  = DataT;
  using BDataType  = DataT;
  using EDataType  = DistT;
  using KernelArgs = PairwiseDistanceCkKernelArgs<DataT, DistT>;

  using EpilogueType =
    ck::PairwiseDistanceCkEpilogue<PairwiseDistanceCkEpilogueProblem<DataT, DistT, ScalarVec>,
                                   ElementWiseOp>;

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

  __host__ static auto IsSupportedArgument(const KernelArgs& kargs) -> bool
  {
    if (kargs.k_batch > 1) return false;
    const auto kargs_adapted = to_universal_args(kargs);
    return HelperKernel::IsSupportedArgument(kargs_adapted);
  }

  __device__ void operator()(KernelArgs kargs) const
  {
    const auto [i_m, i_n]     = get_tile_indices(kargs);
    const auto universal_args = to_universal_args(kargs);
    typename HelperKernel::SplitKBatchOffset splitk_batch_offset(universal_args);

    __shared__ char smem_ptr_0[GetSmemSize()];
    const auto gemm_pointers = make_gemm_pointers(kargs, splitk_batch_offset);

    const auto& gemm_tensor_views_tuple =
      HelperKernel::template MakeGemmTensorViews<DummyEpilogue::MemoryOperation>(
        gemm_pointers.as_ptr,
        gemm_pointers.bs_ptr,
        gemm_pointers.ds_ptr,
        gemm_pointers.e_ptr,
        universal_args,
        splitk_batch_offset.splitted_k);

    const auto& gemm_pad_views = HelperKernel::MakeGemmPadViews(gemm_tensor_views_tuple);
    auto gemm_tile_windows     = HelperKernel::MakeGemmTileWindows(gemm_pad_views, i_m, i_n);

    const ck_tile::index_t num_loop = ck_tile::amd_wave_read_first_lane(
      TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

    const auto& as_block_window = gemm_tile_windows.at(ck_tile::number<0>{});
    const auto& bs_block_window = gemm_tile_windows.at(ck_tile::number<1>{});
    auto& e_block_window        = gemm_tile_windows.at(ck_tile::number<3>{});

    run_gemm_and_epilogue(
      as_block_window, bs_block_window, e_block_window, num_loop, smem_ptr_0, kargs, i_m, i_n);
  }

 private:
  // --- Helpers for operator() ---
  __device__ static std::tuple<ck_tile::index_t, ck_tile::index_t> get_tile_indices(
    const KernelArgs& kargs)
  {
    const auto blockId         = ck_tile::amd_wave_read_first_lane(blockIdx.x);
    const auto [iM, iN]        = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
    const ck_tile::index_t i_m = ck_tile::amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
    const ck_tile::index_t i_n = ck_tile::amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
    return {i_m, i_n};
  }

  // [MODIFIED vs 19_gemm_multi_d] ds_ptr empty; norm data passed to epilogue separately
  struct GemmPointers {
    std::array<const ADataType*, 1> as_ptr;
    std::array<const BDataType*, 1> bs_ptr;
    std::array<const void*, 0> ds_ptr;
    EDataType* e_ptr;
  };

  __device__ static GemmPointers make_gemm_pointers(
    const KernelArgs& kargs, const typename HelperKernel::SplitKBatchOffset& splitk_batch_offset)
  {
    return {
      {kargs.a_ptr + splitk_batch_offset.as_k_split_offset[0]},
      {kargs.b_ptr + splitk_batch_offset.bs_k_split_offset[0]},
      {},
      kargs.e_ptr,
    };
  }

  template <typename AsWindowT, typename BsWindowT, typename EWindowT>
  __device__ void run_gemm_and_epilogue(const AsWindowT& as_block_window,
                                        const BsWindowT& bs_block_window,
                                        EWindowT& e_block_window,
                                        ck_tile::index_t num_loop,
                                        void* smem_ptr,
                                        const KernelArgs& kargs,
                                        ck_tile::index_t i_m,
                                        ck_tile::index_t i_n) const
  {
    const auto& c_block_tile =
      GemmPipeline{}.template operator()(as_block_window,
                                         typename GemmPipeline::AElementWise{},
                                         bs_block_window,
                                         typename GemmPipeline::BElementWise{},
                                         num_loop,
                                         smem_ptr);
    // Custom epilogue: norm_x_ptr, norm_y_ptr instead of D tensor windows
    EpilogueType{}(e_block_window,
                   c_block_tile,
                   smem_ptr,
                   kargs.norm_x_ptr,
                   kargs.norm_y_ptr,
                   i_m,
                   i_n,
                   kargs.M,
                   kargs.N);
  }

  // UniversalGemmKernelArgs<1,1,0>: 0 D tensors (multiD uses 1,1,N for N D tensors)
  __host__ __device__ static constexpr ck_tile::UniversalGemmKernelArgs<1, 1, 0> to_universal_args(
    const KernelArgs& k)
  {
    return ck_tile::UniversalGemmKernelArgs<1, 1, 0>{
      {k.a_ptr},
      {k.b_ptr},
      {},
      k.e_ptr,
      k.M,
      k.N,
      k.K,
      {k.stride_A},
      {k.stride_B},
      {},
      k.stride_E,
      k.k_batch,
    };
  }
};

}  // namespace cuvs::distance::detail
