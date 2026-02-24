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

// Generic norm-based standalone kernel: GEMM + epilogue using norm_x/norm_y
// directly, without materializing the m×n D0 tensor. Parameterized by
// ElementWiseOp (e.g. CosineDistance, L2Expanded) which takes (e, c, norm_x, norm_y).

#pragma once

#include "ck_norm_based_epilogue.hpp"

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp>

namespace ck_tile {

// Dummy epilogue with NumDTensor=0 for tensor view creation. Never actually invoked.
template <typename DataT, typename DistT, bool ScalarVec>
struct PairwiseDistanceCkStandaloneDummyEpilogue {
  using ODataType                                        = DistT;
  using DsDataType                                       = tuple<>;
  using DsLayout                                         = tuple<>;
  static constexpr memory_operation_enum MemoryOperation = memory_operation_enum::set;

  __host__ __device__ static constexpr index_t GetVectorSizeC()
  {
    return ScalarVec ? 1 : std::min(4, static_cast<int>(16 / sizeof(ODataType)));
  }
  template <index_t I>
  __host__ __device__ static constexpr index_t GetVectorSizeD(number<I>)
  {
    return 1;
  }
  __host__ __device__ static constexpr index_t GetSmemSize() { return 0; }
};

// Kernel args for norm-based standalone: no D0, uses norm_x and norm_y.
template <typename DataT, typename DistT>
struct PairwiseDistanceCkStandaloneKernelArgs {
  const DataT* a_ptr;
  const DataT* b_ptr;
  DistT* e_ptr;
  const DistT* norm_x_ptr;
  const DistT* norm_y_ptr;
  index_t M;
  index_t N;
  index_t K;
  index_t stride_A;
  index_t stride_B;
  index_t stride_E;
  index_t k_batch;
};

// Epilogue problem matching CShuffleEpilogue layout.
template <typename DataT, typename DistT, bool ScalarVec>
using PairwiseDistanceCkStandaloneEpilogueProblem =
  PairwiseDistanceCkEpilogueProblem<float,
                                    DistT,
                                    GemmShape::kM,
                                    GemmShape::kN,
                                    GemmShape::BlockWarps::at(number<0>{}),
                                    GemmShape::BlockWarps::at(number<1>{}),
                                    GemmShape::WarpTile::at(number<0>{}),
                                    GemmShape::WarpTile::at(number<1>{}),
                                    GemmShape::WarpTile::at(number<2>{}),
                                    ScalarVec,
                                    1,
                                    1>;

// Generic standalone kernel: ElementWiseOp(e, c, norm_x, norm_y).
template <typename DataT,
          typename DistT,
          typename InputLayout,
          bool ScalarVec,
          typename ElementWiseOp>
struct PairwiseDistanceCkStandaloneKernel {
  using GemmPipeline    = DistanceGemmPipeline<DataT, DistT, InputLayout, ScalarVec>;
  using TilePartitioner = ck_tile::TilePartitioner;
  using DummyEpilogue   = PairwiseDistanceCkStandaloneDummyEpilogue<DataT, DistT, ScalarVec>;
  using HelperKernel    = UniversalGemmKernel<TilePartitioner, GemmPipeline, DummyEpilogue>;

  using ADataType  = DataT;
  using BDataType  = DataT;
  using EDataType  = DistT;
  using KernelArgs = PairwiseDistanceCkStandaloneKernelArgs<DataT, DistT>;

  static constexpr index_t kBlockSize = HelperKernel::kBlockSize;

  using EpilogueType =
    PairwiseDistanceCkEpilogue<PairwiseDistanceCkStandaloneEpilogueProblem<DataT, DistT, ScalarVec>,
                               ElementWiseOp>;

  __host__ static constexpr auto GridSize(index_t M, index_t N, index_t KBatch) -> dim3
  {
    return HelperKernel::GridSize(M, N, KBatch);
  }

  __host__ static constexpr auto BlockSize() -> dim3 { return HelperKernel::BlockSize(); }

  __host__ static constexpr index_t GetSmemSize()
  {
    return max(GemmPipeline::GetSmemSize(), EpilogueType::GetSmemSize());
  }

  __host__ static auto IsSupportedArgument(const KernelArgs& kargs) -> bool
  {
    if (kargs.k_batch > 1) return false;
    const auto kargs_adapted = to_universal_args(kargs);
    return HelperKernel::IsSupportedArgument(kargs_adapted);
  }

  __device__ void operator()(KernelArgs kargs) const
  {
    const auto blockId  = amd_wave_read_first_lane(blockIdx.x);
    const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
    const index_t i_m   = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
    const index_t i_n   = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

    typename HelperKernel::SplitKBatchOffset splitk_batch_offset(to_universal_args(kargs));

    std::array<const ADataType*, 1> as_ptr{
      {kargs.a_ptr + splitk_batch_offset.as_k_split_offset[0]}};
    std::array<const BDataType*, 1> bs_ptr{
      {kargs.b_ptr + splitk_batch_offset.bs_k_split_offset[0]}};
    std::array<const void*, 0> ds_ptr{};
    EDataType* e_ptr = kargs.e_ptr;

    __shared__ char smem_ptr_0[GetSmemSize()];

    const auto& gemm_tensor_views_tuple =
      HelperKernel::template MakeGemmTensorViews<DummyEpilogue::MemoryOperation>(
        as_ptr, bs_ptr, ds_ptr, e_ptr, to_universal_args(kargs), splitk_batch_offset.splitted_k);

    const auto& gemm_pad_views = HelperKernel::MakeGemmPadViews(gemm_tensor_views_tuple);
    auto gemm_tile_windows     = HelperKernel::MakeGemmTileWindows(gemm_pad_views, i_m, i_n);
    const index_t num_loop =
      amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

    const auto& as_block_window = gemm_tile_windows.at(number<0>{});
    const auto& bs_block_window = gemm_tile_windows.at(number<1>{});
    auto& e_block_window        = gemm_tile_windows.at(number<3>{});

    const auto& c_block_tile =
      GemmPipeline{}.template operator()(as_block_window,
                                         typename GemmPipeline::AElementWise{},
                                         bs_block_window,
                                         typename GemmPipeline::BElementWise{},
                                         num_loop,
                                         smem_ptr_0);

    EpilogueType{}(e_block_window,
                   c_block_tile,
                   smem_ptr_0,
                   kargs.norm_x_ptr,
                   kargs.norm_y_ptr,
                   i_m,
                   i_n,
                   kargs.M,
                   kargs.N);
  }

 private:
  __host__ __device__ static constexpr UniversalGemmKernelArgs<1, 1, 0> to_universal_args(
    const KernelArgs& k)
  {
    return UniversalGemmKernelArgs<1, 1, 0>{
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

}  // namespace ck_tile
