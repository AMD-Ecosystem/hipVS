// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../test_utils.cuh"

#include <distance/fused_distance_nn-inl.cuh>

#include <cuvs/distance/distance.hpp>
#include <raft/common/nvtx.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <limits>

namespace raft {
template <typename IdxT, typename DataT>
std::ostream& operator<<(std::ostream& os, const KeyValuePair<IdxT, DataT>& kvp)
{
  os << "{key=" << kvp.key << ", value=" << kvp.value << "}";
  return os;
}
}  // namespace raft

namespace cuvs {
namespace distance {

template <typename DataT, typename IdxT = int>
struct FusedDistanceNNInputs {
  DataT tolerance;
  IdxT m;
  IdxT n;
  IdxT k;
  bool sqrt;
  cuvs::distance::DistanceType metric;
  unsigned long long seed;
};

template <typename DataT, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const FusedDistanceNNInputs<DataT, IdxT>& p)
{
  os << "m=" << p.m << ", n=" << p.n << ", k=" << p.k << ", sqrt=" << p.sqrt
     << ", metric=" << static_cast<int>(p.metric) << ", seed=" << p.seed;
  return os;
}

template <typename DataT, typename IdxT>
RAFT_KERNEL naiveFusedL2NNKernel(raft::KeyValuePair<IdxT, DataT>* min,
                                 const DataT* x,
                                 const DataT* y,
                                 const DataT* xn,
                                 const DataT* yn,
                                 IdxT m,
                                 IdxT n,
                                 IdxT k,
                                 bool sqrt)
{
  IdxT midx = threadIdx.x + blockIdx.x * blockDim.x;
  if (midx >= m) return;

  DataT min_dist  = std::numeric_limits<DataT>::max();
  IdxT min_idx    = 0;
  DataT x_norm_sq = xn[midx];

  for (IdxT nidx = 0; nidx < n; ++nidx) {
    DataT acc = DataT(0);
    for (IdxT i = 0; i < k; ++i) {
      DataT diff = x[midx * k + i] - y[nidx * k + i];
      acc += diff * diff;
    }

    if (sqrt) { acc = raft::sqrt(acc); }

    if (acc < min_dist) {
      min_dist = acc;
      min_idx  = nidx;
    }
  }

  min[midx].key   = min_idx;
  min[midx].value = min_dist;
}

template <typename DataT, typename IdxT>
RAFT_KERNEL naiveFusedCosineNNKernel(raft::KeyValuePair<IdxT, DataT>* min,
                                     const DataT* x,
                                     const DataT* y,
                                     const DataT* xn,
                                     const DataT* yn,
                                     IdxT m,
                                     IdxT n,
                                     IdxT k,
                                     bool sqrt)
{
  IdxT midx = threadIdx.x + blockIdx.x * blockDim.x;
  if (midx >= m) return;

  DataT min_dist = std::numeric_limits<DataT>::max();
  IdxT min_idx   = 0;
  DataT x_norm   = xn[midx];

  for (IdxT nidx = 0; nidx < n; ++nidx) {
    DataT dot_product = DataT(0);
    for (IdxT i = 0; i < k; ++i) {
      dot_product += x[midx * k + i] * y[nidx * k + i];
    }

    DataT y_norm   = yn[nidx];
    DataT cos_dist = DataT(1) - (dot_product / (x_norm * y_norm));

    if (cos_dist < min_dist) {
      min_dist = cos_dist;
      min_idx  = nidx;
    }
  }

  min[midx].key   = min_idx;
  min[midx].value = min_dist;
}

template <typename DataT, typename IdxT>
void naiveFusedDistanceNN(raft::KeyValuePair<IdxT, DataT>* min,
                          const DataT* x,
                          const DataT* y,
                          const DataT* xn,
                          const DataT* yn,
                          IdxT m,
                          IdxT n,
                          IdxT k,
                          bool sqrt,
                          cuvs::distance::DistanceType metric,
                          cudaStream_t stream)
{
  static const int TPB = 256;
  int nblks            = raft::ceildiv(m, (IdxT)TPB);

  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      naiveFusedL2NNKernel<DataT, IdxT>
        <<<nblks, TPB, 0, stream>>>(min, x, y, xn, yn, m, n, k, sqrt);
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      naiveFusedCosineNNKernel<DataT, IdxT>
        <<<nblks, TPB, 0, stream>>>(min, x, y, xn, yn, m, n, k, sqrt);
      break;
    default: FAIL() << "Unsupported metric for naiveFusedDistanceNN";
  }
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT = int>
class FusedDistanceNNTest : public ::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>> {
 public:
  FusedDistanceNNTest()
    : params(::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      xn(params.m, stream),
      yn(params.n, stream),
      min_ref(params.m, stream),
      min_actual(params.m, stream),
      workspace(params.m, stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState rng(params.seed);
    IdxT m = params.m;
    IdxT n = params.n;
    IdxT k = params.k;

    uniform(handle, rng, x.data(), m * k, DataT(-1.0), DataT(1.0));
    uniform(handle, rng, y.data(), n * k, DataT(-1.0), DataT(1.0));

    computeNorms();

    naiveFusedDistanceNN(min_ref.data(),
                         x.data(),
                         y.data(),
                         xn.data(),
                         yn.data(),
                         m,
                         n,
                         k,
                         params.sqrt,
                         params.metric,
                         stream);

    fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(min_actual.data(),
                                                                           x.data(),
                                                                           y.data(),
                                                                           xn.data(),
                                                                           yn.data(),
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           workspace.data(),
                                                                           params.sqrt,
                                                                           true,
                                                                           true,
                                                                           params.metric,
                                                                           0.0f,
                                                                           stream);

    raft::resource::sync_stream(handle, stream);
  }

  void computeNorms()
  {
    IdxT m = params.m;
    IdxT n = params.n;
    IdxT k = params.k;

    if (params.metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        xn.data(), x.data(), k, m, stream, raft::sqrt_op{});
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        yn.data(), y.data(), k, n, stream, raft::sqrt_op{});
    } else {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(xn.data(), x.data(), k, m, stream);
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(yn.data(), y.data(), k, n, stream);
    }
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  FusedDistanceNNInputs<DataT, IdxT> params;
  rmm::device_uvector<DataT> x, y, xn, yn;
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_ref, min_actual;
  rmm::device_uvector<int> workspace;
};

template <typename T>
struct KVPCompare {
  cuvs::CompareApprox<T> compare;
  KVPCompare(T tolerance) : compare{tolerance} {}

  template <typename IdxT>
  bool operator()(const raft::KeyValuePair<IdxT, T>& a, const raft::KeyValuePair<IdxT, T>& b) const
  {
    if (a.key != b.key) { return false; }
    return compare(a.value, b.value);
  }
};

template <typename DataT, typename IdxT>
RAFT_KERNEL naiveFusedL2NNKernelMinOnly(DataT* min,
                                        const DataT* x,
                                        const DataT* y,
                                        const DataT* xn,
                                        const DataT* yn,
                                        IdxT m,
                                        IdxT n,
                                        IdxT k,
                                        bool sqrt)
{
  IdxT midx = threadIdx.x + blockIdx.x * blockDim.x;
  if (midx >= m) return;

  DataT min_dist = std::numeric_limits<DataT>::max();

  for (IdxT nidx = 0; nidx < n; ++nidx) {
    DataT acc = DataT(0);
    for (IdxT i = 0; i < k; ++i) {
      DataT diff = x[midx * k + i] - y[nidx * k + i];
      acc += diff * diff;
    }

    if (sqrt) { acc = raft::sqrt(acc); }

    if (acc < min_dist) { min_dist = acc; }
  }

  min[midx] = min_dist;
}

template <typename DataT, typename IdxT>
void naiveFusedDistanceNNMinOnly(DataT* min,
                                 const DataT* x,
                                 const DataT* y,
                                 const DataT* xn,
                                 const DataT* yn,
                                 IdxT m,
                                 IdxT n,
                                 IdxT k,
                                 bool sqrt,
                                 cudaStream_t stream)
{
  static const int TPB = 256;
  int nblks            = raft::ceildiv(m, (IdxT)TPB);
  naiveFusedL2NNKernelMinOnly<DataT, IdxT>
    <<<nblks, TPB, 0, stream>>>(min, x, y, xn, yn, m, n, k, sqrt);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT = int>
class FusedDistanceNNMinOnlyTest
  : public ::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>> {
 public:
  FusedDistanceNNMinOnlyTest()
    : params(::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      xn(params.m, stream),
      yn(params.n, stream),
      min_ref(params.m, stream),
      min_actual(params.m, stream),
      workspace(params.m, stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState rng(params.seed);
    IdxT m = params.m;
    IdxT n = params.n;
    IdxT k = params.k;

    uniform(handle, rng, x.data(), m * k, DataT(-1.0), DataT(1.0));
    uniform(handle, rng, y.data(), n * k, DataT(-1.0), DataT(1.0));

    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(xn.data(), x.data(), k, m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(yn.data(), y.data(), k, n, stream);

    naiveFusedDistanceNNMinOnly(
      min_ref.data(), x.data(), y.data(), xn.data(), yn.data(), m, n, k, params.sqrt, stream);

    fusedDistanceNNMinReduce<DataT, DataT, IdxT>(min_actual.data(),
                                                 x.data(),
                                                 y.data(),
                                                 xn.data(),
                                                 yn.data(),
                                                 m,
                                                 n,
                                                 k,
                                                 workspace.data(),
                                                 params.sqrt,
                                                 true,
                                                 true,
                                                 params.metric,
                                                 0.0f,
                                                 stream);

    raft::resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  FusedDistanceNNInputs<DataT, IdxT> params;
  rmm::device_uvector<DataT> x, y, xn, yn;
  rmm::device_uvector<DataT> min_ref, min_actual;
  rmm::device_uvector<int> workspace;
};

// ======================= Float Tests =======================

using FusedDistanceNNTestF = FusedDistanceNNTest<float, int>;

TEST_P(FusedDistanceNNTestF, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int>> inputsF = {
  // Basic L2 tests
  {0.01f, 32, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 64, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 128, 128, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 256, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},

  // L2 sqrt tests
  {0.01f, 64, 64, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  {0.01f, 128, 128, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},

  // Skinny matrices (k < 32)
  {0.01f, 64, 64, 8, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {0.01f, 128, 128, 16, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {0.01f, 256, 256, 24, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},

  // Asymmetric dimensions
  {0.01f, 128, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {0.01f, 64, 128, 32, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {0.01f, 256, 64, 48, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},

  // Larger tests
  {0.01f, 512, 512, 64, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
  {0.01f, 1024, 1024, 32, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
  {0.01f, 1024, 512, 64, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},

  // Cosine distance tests
  {0.01f, 64, 64, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
  {0.01f, 128, 128, 64, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
  {0.01f, 256, 256, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},

  // Alignment edge cases
  {0.01f, 33, 33, 33, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {0.01f, 65, 65, 17, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {0.01f, 129, 127, 63, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNTests, FusedDistanceNNTestF, ::testing::ValuesIn(inputsF));

// ======================= Double Tests =======================

using FusedDistanceNNTestD = FusedDistanceNNTest<double, int>;

TEST_P(FusedDistanceNNTestD, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<double>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<double, int>> inputsD = {
  {0.001, 64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 128, 128, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 64, 64, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  {0.001, 128, 128, 16, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {0.001, 64, 64, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNTests, FusedDistanceNNTestD, ::testing::ValuesIn(inputsD));

#ifdef CUVS_CK_ENABLED
// ======================= CK Skinny-K Optimization Tests =======================
// These tests focus on the skinny-K optimization boundary (K < 32) to validate
// both correctness and performance improvements for small feature dimensions
// commonly seen in k-means clustering workloads.

using FusedDistanceNNCkSkinnyTest = FusedDistanceNNTest<float, int>;

TEST_P(FusedDistanceNNCkSkinnyTest, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int>> inputsCkSkinny = {
  // Skinny-K optimization boundary tests (K < 32 threshold)
  {0.01f, 64, 64, 2, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // Very small K
  {0.01f, 64, 64, 4, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=4 (skinny
                                                                                 // config)
  {0.01f, 64, 64, 8, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=8 (optimal for
                                                                                 // fp32 skinny)
  {0.01f, 64, 64, 16, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=16 (skinny
                                                                                  // config)
  {0.01f, 64, 64, 24, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=24 (still
                                                                                  // skinny)
  {0.01f, 64, 64, 31, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=31 (edge
                                                                                  // case, skinny)
  {0.01f, 64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=32 (boundary,
                                                                                  // default config)
  {0.01f, 64, 64, 33, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},  // K=33 (default
                                                                                  // config)

  // Test both metrics with skinny K
  {0.01f, 64, 64, 8, false, cuvs::distance::DistanceType::CosineExpanded, 5678ULL},  // Cosine with
                                                                                     // skinny K
  {0.01f, 64, 64, 16, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},  // L2 sqrt with
                                                                                     // skinny K

  // Mixed tile sizes with skinny K (typical k-means scenarios)
  {0.01f, 32, 128, 8, false, cuvs::distance::DistanceType::L2Expanded, 2468ULL},  // Fewer samples,
                                                                                  // more centroids
  {0.01f,
   128,
   32,
   16,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   3579ULL},  // More samples, fewer centroids
  {0.01f, 256, 64, 12, false, cuvs::distance::DistanceType::L2Expanded, 4680ULL},  // Large batch
                                                                                   // with skinny K

  // Legacy vector alignment tests - maintained for compatibility
  {0.01f, 64, 64, 1, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},  // K=1 (minimal)
  {0.01f, 64, 64, 3, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},  // K=3
                                                                                 // (non-aligned)
  {0.01f, 64, 64, 5, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},  // K=5
                                                                                 // (non-aligned)
  {0.01f, 64, 64, 7, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},  // K=7
                                                                                 // (non-aligned)

  // Edge cases that stress different code paths
  {0.01f, 1, 1, 1, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},   // 1x1x1 minimum
  {0.01f, 1, 64, 4, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},  // Single row with
                                                                                // skinny K
  {0.01f, 64, 1, 8, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},  // Single column
                                                                                // with skinny K
  {0.01f, 1, 1024, 16, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},  // Very wide with
                                                                                   // skinny K

  // Mixed scenarios combining skinny K with various tile alignments
  {0.01f, 33, 65, 6, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},  // Odd M,N with
                                                                                 // skinny K
  {0.01f,
   96,
   32,
   12,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   3456ULL},  // Mixed tile alignment with skinny K

  // Large-M stress tests with skinny K (realistic k-means scenarios - scaled for test environment)
  {0.01f, 100000, 64, 2, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},  // 100K points,
                                                                                     // 2D features
  {0.01f, 200000, 32, 4, false, cuvs::distance::DistanceType::L2Expanded, 8901ULL},  // 200K points,
                                                                                     // 4D features
  {0.01f,
   150000,
   128,
   8,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   9012ULL},  // 150K points, 8D features
  {0.01f,
   80000,
   256,
   16,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   1123ULL},  // 80K points, 16D features
  {0.01f,
   500000,
   16,
   12,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   2234ULL},  // 500K points, 12D features

  // Large-M with cosine distance (scaled for test environment)
  {0.01f,
   120000,
   100,
   24,
   false,
   cuvs::distance::DistanceType::CosineExpanded,
   3345ULL},  // 120K documents, 24D embeddings
  {0.01f,
   250000,
   50,
   16,
   false,
   cuvs::distance::DistanceType::CosineExpanded,
   4456ULL}  // 250K documents, 16D embeddings
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNTests,
                        FusedDistanceNNCkSkinnyTest,
                        ::testing::ValuesIn(inputsCkSkinny));
#endif  // CUVS_CK_ENABLED

// ======================= Int64 Index Tests =======================

using FusedDistanceNNTestF_Int64 = FusedDistanceNNTest<float, int64_t>;

TEST_P(FusedDistanceNNTestF_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int64_t>> inputsF_Int64 = {
  {0.01f, 64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 128, 128, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 256, 256, 32, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNTests,
                        FusedDistanceNNTestF_Int64,
                        ::testing::ValuesIn(inputsF_Int64));

// ======================= Double + Int64 Index Tests =======================

using FusedDistanceNNTestD_Int64 = FusedDistanceNNTest<double, int64_t>;

TEST_P(FusedDistanceNNTestD_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<double>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<double, int64_t>> inputsD_Int64 = {
  {0.001, 64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 128, 128, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 64, 64, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  {0.001, 64, 64, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNTests,
                        FusedDistanceNNTestD_Int64,
                        ::testing::ValuesIn(inputsD_Int64));

// ======================= Min-Only Output Tests (for inertia computation) =======================

using FusedDistanceNNMinOnlyTestF = FusedDistanceNNMinOnlyTest<float, int>;

TEST_P(FusedDistanceNNMinOnlyTestF, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref.data(),
                                min_actual.data(),
                                params.m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<float, int>> inputsMinOnlyF = {
  // Basic tests
  {0.01f, 64, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 128, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 256, 128, 32, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
  // L2 sqrt
  {0.01f, 128, 64, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  // Kmeans-like dimensions
  {0.01f, 1000, 32, 128, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {0.01f, 2000, 64, 256, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNMinOnlyTests,
                        FusedDistanceNNMinOnlyTestF,
                        ::testing::ValuesIn(inputsMinOnlyF));

// ======================= Min-Only Double Tests =======================

using FusedDistanceNNMinOnlyTestD = FusedDistanceNNMinOnlyTest<double, int>;

TEST_P(FusedDistanceNNMinOnlyTestD, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref.data(),
                                min_actual.data(),
                                params.m,
                                cuvs::CompareApprox<double>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<double, int>> inputsMinOnlyD = {
  {0.001, 64, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 128, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 128, 64, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNMinOnlyTests,
                        FusedDistanceNNMinOnlyTestD,
                        ::testing::ValuesIn(inputsMinOnlyD));

// ======================= Min-Only Float Int64 Tests =======================

using FusedDistanceNNMinOnlyTestF_Int64 = FusedDistanceNNMinOnlyTest<float, int64_t>;

TEST_P(FusedDistanceNNMinOnlyTestF_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref.data(),
                                min_actual.data(),
                                params.m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<float, int64_t>> inputsMinOnlyF_Int64 = {
  {0.01f, 64, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 128, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 128, 64, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNMinOnlyTests,
                        FusedDistanceNNMinOnlyTestF_Int64,
                        ::testing::ValuesIn(inputsMinOnlyF_Int64));

// ======================= Min-Only Double Int64 Tests =======================

using FusedDistanceNNMinOnlyTestD_Int64 = FusedDistanceNNMinOnlyTest<double, int64_t>;

TEST_P(FusedDistanceNNMinOnlyTestD_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref.data(),
                                min_actual.data(),
                                params.m,
                                cuvs::CompareApprox<double>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<double, int64_t>> inputsMinOnlyD_Int64 = {
  {0.001, 64, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 128, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 128, 64, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNMinOnlyTests,
                        FusedDistanceNNMinOnlyTestD_Int64,
                        ::testing::ValuesIn(inputsMinOnlyD_Int64));

// ======================= Realistic Kmeans-style Tests =======================
// These test dimensions commonly seen in kmeans clustering:
// - m (samples): batch sizes like 1000, 2000, 4096
// - n (clusters): 8, 16, 32, 64, 128, 256, 512
// - k (features): embedding dimensions like 32, 64, 128, 256, 512

using FusedDistanceNNKmeansTestF = FusedDistanceNNTest<float, int>;

TEST_P(FusedDistanceNNKmeansTestF, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int>> inputsKmeansF = {
  // Small cluster counts (typical early iterations or small datasets)
  // m=samples, n=clusters, k=features
  {0.01f, 1000, 8, 64, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {0.01f, 1000, 16, 64, false, cuvs::distance::DistanceType::L2Expanded, 2222ULL},
  {0.01f, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 3333ULL},

  // Medium cluster counts
  {0.01f, 2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 4444ULL},
  {0.01f, 2000, 128, 128, false, cuvs::distance::DistanceType::L2Expanded, 5555ULL},
  {0.01f, 4096, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 6666ULL},

  // Large cluster counts (hierarchical kmeans scenarios)
  {0.01f, 4096, 512, 32, false, cuvs::distance::DistanceType::L2Expanded, 7777ULL},
  {0.01f, 8192, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 8888ULL},

  // High-dimensional embeddings
  {0.01f, 1000, 32, 256, false, cuvs::distance::DistanceType::L2Expanded, 9999ULL},
  {0.01f, 1000, 32, 512, false, cuvs::distance::DistanceType::L2Expanded, 1010ULL},
  {0.01f, 2000, 64, 384, false, cuvs::distance::DistanceType::L2Expanded, 1212ULL},

  // L2 sqrt (used when metric != L2Expanded)
  {0.01f, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1313ULL},
  {0.01f, 2000, 64, 256, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1414ULL},

  // Cosine distance (used in some kmeans variants)
  {0.01f, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 1515ULL},
  {0.01f, 2000, 64, 256, false, cuvs::distance::DistanceType::CosineExpanded, 1616ULL},

  // Skinny matrices (k < 32, triggers different policy)
  {0.01f, 1000, 64, 16, false, cuvs::distance::DistanceType::L2Expanded, 1717ULL},
  {0.01f, 2000, 128, 24, false, cuvs::distance::DistanceType::L2Expanded, 1818ULL},

  // Edge cases: very few clusters
  {0.01f, 5000, 2, 128, false, cuvs::distance::DistanceType::L2Expanded, 1919ULL},
  {0.01f, 5000, 4, 128, false, cuvs::distance::DistanceType::L2Expanded, 2020ULL},

  // Edge cases: many clusters relative to samples
  {0.01f, 512, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 2121ULL},
  {0.01f, 1024, 512, 64, false, cuvs::distance::DistanceType::L2Expanded, 2323ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNKmeansTests,
                        FusedDistanceNNKmeansTestF,
                        ::testing::ValuesIn(inputsKmeansF));

// ======================= Realistic Kmeans-style Double Tests =======================

using FusedDistanceNNKmeansTestD = FusedDistanceNNTest<double, int>;

TEST_P(FusedDistanceNNKmeansTestD, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<double>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<double, int>> inputsKmeansD = {
  {0.001, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {0.001, 2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 4444ULL},
  {0.001, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1313ULL},
  {0.001, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 1515ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNKmeansTests,
                        FusedDistanceNNKmeansTestD,
                        ::testing::ValuesIn(inputsKmeansD));

// ======================= Realistic Kmeans-style Float Int64 Tests =======================

using FusedDistanceNNKmeansTestF_Int64 = FusedDistanceNNTest<float, int64_t>;

TEST_P(FusedDistanceNNKmeansTestF_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int64_t>> inputsKmeansF_Int64 = {
  {0.01f, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {0.01f, 2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 4444ULL},
  {0.01f, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1313ULL},
  {0.01f, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 1515ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNKmeansTests,
                        FusedDistanceNNKmeansTestF_Int64,
                        ::testing::ValuesIn(inputsKmeansF_Int64));

// ======================= Realistic Kmeans-style Double Int64 Tests =======================

using FusedDistanceNNKmeansTestD_Int64 = FusedDistanceNNTest<double, int64_t>;

TEST_P(FusedDistanceNNKmeansTestD_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<double>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<double, int64_t>> inputsKmeansD_Int64 = {
  {0.001, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {0.001, 2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 4444ULL},
  {0.001, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1313ULL},
  {0.001, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 1515ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNKmeansTests,
                        FusedDistanceNNKmeansTestD_Int64,
                        ::testing::ValuesIn(inputsKmeansD_Int64));

// ======================= Pre-initialized Buffer Tests =======================
// Tests with initOutBuffer=false, simulating the real kmeans usage where
// the output buffer is pre-filled with max values

template <typename DataT, typename IdxT = int>
class FusedDistanceNNPreInitTest
  : public ::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>> {
 public:
  FusedDistanceNNPreInitTest()
    : params(::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      xn(params.m, stream),
      yn(params.n, stream),
      min_ref(params.m, stream),
      min_actual(params.m, stream),
      workspace(params.m, stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState rng(params.seed);
    IdxT m = params.m;
    IdxT n = params.n;
    IdxT k = params.k;

    raft::random::uniform(handle, rng, x.data(), m * k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(handle, rng, y.data(), n * k, DataT(-1.0), DataT(1.0));

    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(xn.data(), x.data(), k, m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(yn.data(), y.data(), k, n, stream);

    naiveFusedDistanceNN(min_ref.data(),
                         x.data(),
                         y.data(),
                         xn.data(),
                         yn.data(),
                         m,
                         n,
                         k,
                         params.sqrt,
                         params.metric,
                         stream);

    // Pre-initialize the output buffer with max values (like kmeans does)
    raft::KeyValuePair<IdxT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    thrust::fill(thrust::device, min_actual.data(), min_actual.data() + m, initial_value);

    // Call with initOutBuffer=false
    fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(
      min_actual.data(),
      x.data(),
      y.data(),
      xn.data(),
      yn.data(),
      m,
      n,
      k,
      workspace.data(),
      params.sqrt,
      false,  // initOutBuffer=false
      true,
      params.metric,
      0.0f,
      stream);

    raft::resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  FusedDistanceNNInputs<DataT, IdxT> params;
  rmm::device_uvector<DataT> x, y, xn, yn;
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_ref, min_actual;
  rmm::device_uvector<int> workspace;
};

using FusedDistanceNNPreInitTestF = FusedDistanceNNPreInitTest<float, int>;

TEST_P(FusedDistanceNNPreInitTestF, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int>> inputsPreInitF = {
  {0.01f, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
  {0.01f, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.01f, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNPreInitTests,
                        FusedDistanceNNPreInitTestF,
                        ::testing::ValuesIn(inputsPreInitF));

// ======================= Pre-init Double Tests =======================

using FusedDistanceNNPreInitTestD = FusedDistanceNNPreInitTest<double, int>;

TEST_P(FusedDistanceNNPreInitTestD, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<double>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<double, int>> inputsPreInitD = {
  {0.001, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.001, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNPreInitTests,
                        FusedDistanceNNPreInitTestD,
                        ::testing::ValuesIn(inputsPreInitD));

// ======================= Pre-init Float Int64 Tests =======================

using FusedDistanceNNPreInitTestF_Int64 = FusedDistanceNNPreInitTest<float, int64_t>;

TEST_P(FusedDistanceNNPreInitTestF_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<float>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<float, int64_t>> inputsPreInitF_Int64 = {
  {0.01f, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.01f, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNPreInitTests,
                        FusedDistanceNNPreInitTestF_Int64,
                        ::testing::ValuesIn(inputsPreInitF_Int64));

// ======================= Pre-init Double Int64 Tests =======================

using FusedDistanceNNPreInitTestD_Int64 = FusedDistanceNNPreInitTest<double, int64_t>;

TEST_P(FusedDistanceNNPreInitTestD_Int64, Result)
{
  ASSERT_TRUE(cuvs::devArrMatch(
    min_ref.data(), min_actual.data(), params.m, KVPCompare<double>(params.tolerance), stream));
}

const std::vector<FusedDistanceNNInputs<double, int64_t>> inputsPreInitD_Int64 = {
  {0.001, 1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.001, 1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNPreInitTests,
                        FusedDistanceNNPreInitTestD_Int64,
                        ::testing::ValuesIn(inputsPreInitD_Int64));

// ======================= Large-Scale Tests =======================
// Tests with m in the millions to validate the kernel at production scale.
// Full naive comparison is infeasible at this size, so correctness is checked
// on the first and last 1024 rows only; the rest of the buffer exercises
// memory addressing and kernel launch logic.

static constexpr int kCheckRows = 1024;

template <typename DataT, typename IdxT = int>
class FusedDistanceNNLargeTest
  : public ::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>> {
 public:
  FusedDistanceNNLargeTest()
    : params(::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      xn(params.m, stream),
      yn(params.n, stream),
      min_actual(params.m, stream),
      workspace(params.m, stream),
      // Reference buffers: first kCheckRows and last kCheckRows rows only
      min_ref_first(kCheckRows, stream),
      min_ref_last(kCheckRows, stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    raft::random::RngState rng(params.seed);
    IdxT m = params.m;
    IdxT n = params.n;
    IdxT k = params.k;

    uniform(handle, rng, x.data(), m * k, DataT(-1.0), DataT(1.0));
    uniform(handle, rng, y.data(), n * k, DataT(-1.0), DataT(1.0));

    // Compute norms for the full dataset
    if (params.metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        xn.data(), x.data(), k, m, stream, raft::sqrt_op{});
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        yn.data(), y.data(), k, n, stream, raft::sqrt_op{});
    } else {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(xn.data(), x.data(), k, m, stream);
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(yn.data(), y.data(), k, n, stream);
    }

    // Run the full large kernel
    fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(min_actual.data(),
                                                                           x.data(),
                                                                           y.data(),
                                                                           xn.data(),
                                                                           yn.data(),
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           workspace.data(),
                                                                           params.sqrt,
                                                                           true,
                                                                           true,
                                                                           params.metric,
                                                                           0.0f,
                                                                           stream);

    // Naive reference on the first kCheckRows rows
    naiveFusedDistanceNN(min_ref_first.data(),
                         x.data(),
                         y.data(),
                         xn.data(),
                         yn.data(),
                         (IdxT)kCheckRows,
                         n,
                         k,
                         params.sqrt,
                         params.metric,
                         stream);

    // Naive reference on the last kCheckRows rows
    // Offset x and xn by (m - kCheckRows) rows; y and yn are shared
    IdxT last_offset = m - (IdxT)kCheckRows;
    naiveFusedDistanceNN(min_ref_last.data(),
                         x.data() + last_offset * k,
                         y.data(),
                         xn.data() + last_offset,
                         yn.data(),
                         (IdxT)kCheckRows,
                         n,
                         k,
                         params.sqrt,
                         params.metric,
                         stream);

    raft::resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  FusedDistanceNNInputs<DataT, IdxT> params;
  rmm::device_uvector<DataT> x, y, xn, yn;
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_actual;
  rmm::device_uvector<int> workspace;
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_ref_first, min_ref_last;
};

using FusedDistanceNNLargeTestF = FusedDistanceNNLargeTest<float, int>;

TEST_P(FusedDistanceNNLargeTestF, FirstRows)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_first.data(),
                                min_actual.data(),
                                kCheckRows,
                                KVPCompare<float>(params.tolerance),
                                stream));
}

TEST_P(FusedDistanceNNLargeTestF, LastRows)
{
  int last_offset = params.m - kCheckRows;
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_last.data(),
                                min_actual.data() + last_offset,
                                kCheckRows,
                                KVPCompare<float>(params.tolerance),
                                stream));
}

// m=samples, n=clusters, k=features
// Memory footprint (float): x = m*k*4 bytes, e.g. 10M*64*4 = 2.56 GB
const std::vector<FusedDistanceNNInputs<float, int>> inputsLargeF = {
  // 1M samples
  {0.01f, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 1 << 20, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
  {0.01f, 1 << 20, 32, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.01f, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},

  // 10M samples — exercises large-m addressing; k kept small to limit VRAM
  {0.01f, 10000000, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {0.01f, 10000000, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 2222ULL},
  {0.01f, 10000000, 32, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 3333ULL},
  {0.01f, 10000000, 32, 32, false, cuvs::distance::DistanceType::CosineExpanded, 4444ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNLargeTests,
                        FusedDistanceNNLargeTestF,
                        ::testing::ValuesIn(inputsLargeF));

// ======================= Large-Scale Double Tests =======================

using FusedDistanceNNLargeTestD = FusedDistanceNNLargeTest<double, int>;

TEST_P(FusedDistanceNNLargeTestD, FirstRows)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_first.data(),
                                min_actual.data(),
                                kCheckRows,
                                KVPCompare<double>(params.tolerance),
                                stream));
}

TEST_P(FusedDistanceNNLargeTestD, LastRows)
{
  int last_offset = params.m - kCheckRows;
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_last.data(),
                                min_actual.data() + last_offset,
                                kCheckRows,
                                KVPCompare<double>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<double, int>> inputsLargeD = {
  {0.001, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 1 << 20, 32, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.001, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNLargeTests,
                        FusedDistanceNNLargeTestD,
                        ::testing::ValuesIn(inputsLargeD));

// ======================= Large-Scale Float Int64 Tests =======================

using FusedDistanceNNLargeTestF_Int64 = FusedDistanceNNLargeTest<float, int64_t>;

TEST_P(FusedDistanceNNLargeTestF_Int64, FirstRows)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_first.data(),
                                min_actual.data(),
                                kCheckRows,
                                KVPCompare<float>(params.tolerance),
                                stream));
}

TEST_P(FusedDistanceNNLargeTestF_Int64, LastRows)
{
  int64_t last_offset = params.m - kCheckRows;
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_last.data(),
                                min_actual.data() + last_offset,
                                kCheckRows,
                                KVPCompare<float>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<float, int64_t>> inputsLargeF_Int64 = {
  {0.01f, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.01f, 1 << 20, 32, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.01f, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNLargeTests,
                        FusedDistanceNNLargeTestF_Int64,
                        ::testing::ValuesIn(inputsLargeF_Int64));

// ======================= Large-Scale Double Int64 Tests =======================

using FusedDistanceNNLargeTestD_Int64 = FusedDistanceNNLargeTest<double, int64_t>;

TEST_P(FusedDistanceNNLargeTestD_Int64, FirstRows)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_first.data(),
                                min_actual.data(),
                                kCheckRows,
                                KVPCompare<double>(params.tolerance),
                                stream));
}

TEST_P(FusedDistanceNNLargeTestD_Int64, LastRows)
{
  int64_t last_offset = params.m - kCheckRows;
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_last.data(),
                                min_actual.data() + last_offset,
                                kCheckRows,
                                KVPCompare<double>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<double, int64_t>> inputsLargeD_Int64 = {
  {0.001, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {0.001, 1 << 20, 32, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {0.001, 1 << 20, 32, 64, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNLargeTests,
                        FusedDistanceNNLargeTestD_Int64,
                        ::testing::ValuesIn(inputsLargeD_Int64));

#ifdef CUVS_CK_ENABLED
// ======================= CK Large-Scale Skinny-K Tests =======================
// Large-scale tests (multi-million samples) with skinny-K optimization (K < 32).
// These tests validate performance and correctness at production scale for workloads
// like k-means clustering. Only the first and last 1024 rows are checked due to
// the computational cost of full naive validation.

using FusedDistanceNNCkLargeSkinnyTest = FusedDistanceNNLargeTest<float, int>;

TEST_P(FusedDistanceNNCkLargeSkinnyTest, FirstRows)
{
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_first.data(),
                                min_actual.data(),
                                kCheckRows,
                                KVPCompare<float>(params.tolerance),
                                stream));
}

TEST_P(FusedDistanceNNCkLargeSkinnyTest, LastRows)
{
  int last_offset = params.m - kCheckRows;
  ASSERT_TRUE(cuvs::devArrMatch(min_ref_last.data(),
                                min_actual.data() + last_offset,
                                kCheckRows,
                                KVPCompare<float>(params.tolerance),
                                stream));
}

const std::vector<FusedDistanceNNInputs<float, int>> inputsCkLargeSkinny = {
  // Large-M stress tests around 5M with skinny K (realistic large-scale k-means scenarios)
  {0.01f,
   4800000,
   64,
   2,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   5567ULL},  // 4.8M points, 2D features
  {0.01f, 5000000, 32, 4, false, cuvs::distance::DistanceType::L2Expanded, 6678ULL},  // 5M points,
                                                                                      // 4D features
  {0.01f,
   5200000,
   128,
   8,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   7789ULL},  // 5.2M points, 8D features
  {0.01f,
   4900000,
   256,
   16,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   8890ULL},  // 4.9M points, 16D features
  {0.01f,
   5100000,
   16,
   12,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   9901ULL},  // 5.1M points, 12D features
  {0.01f,
   4700000,
   48,
   6,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   1012ULL},  // 4.7M points, 6D features
  {0.01f,
   5300000,
   80,
   24,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   1123ULL},  // 5.3M points, 24D features
  {0.01f,
   4600000,
   96,
   10,
   false,
   cuvs::distance::DistanceType::L2Expanded,
   1234ULL},  // 4.6M points, 10D features

  // Large-M around 5M with cosine distance and skinny K (document embedding scenarios)
  {0.01f,
   5000000,
   64,
   8,
   false,
   cuvs::distance::DistanceType::CosineExpanded,
   1345ULL},  // 5M documents, 8D embeddings
  {0.01f,
   4800000,
   32,
   16,
   false,
   cuvs::distance::DistanceType::CosineExpanded,
   1456ULL},  // 4.8M documents, 16D embeddings
  {0.01f,
   5200000,
   96,
   12,
   false,
   cuvs::distance::DistanceType::CosineExpanded,
   1567ULL},  // 5.2M documents, 12D embeddings

  // Large-M around 5M with L2 sqrt and skinny K (special distance metric scenarios)
  {0.01f,
   5000000,
   128,
   4,
   true,
   cuvs::distance::DistanceType::L2SqrtExpanded,
   1678ULL},  // 5M points, 4D features with sqrt
  {0.01f,
   4900000,
   64,
   20,
   true,
   cuvs::distance::DistanceType::L2SqrtExpanded,
   1789ULL}  // 4.9M points, 20D features with sqrt
};

INSTANTIATE_TEST_CASE_P(FusedDistanceNNCkLargeSkinnyTests,
                        FusedDistanceNNCkLargeSkinnyTest,
                        ::testing::ValuesIn(inputsCkLargeSkinny));
#endif  // CUVS_CK_ENABLED

}  // namespace distance
}  // namespace cuvs
