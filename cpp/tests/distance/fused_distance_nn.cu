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

// ======================= Type-erased shape config =======================

struct ShapeConfig {
  int64_t m;
  int64_t n;
  int64_t k;
  bool sqrt;
  cuvs::distance::DistanceType metric;
  unsigned long long seed;
};

template <typename DataT, typename IdxT>
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
std::vector<FusedDistanceNNInputs<DataT, IdxT>> makeInputs(DataT tolerance,
                                                           const std::vector<ShapeConfig>& shapes)
{
  std::vector<FusedDistanceNNInputs<DataT, IdxT>> out;
  out.reserve(shapes.size());
  for (auto& s : shapes) {
    out.push_back({tolerance,
                   static_cast<IdxT>(s.m),
                   static_cast<IdxT>(s.n),
                   static_cast<IdxT>(s.k),
                   s.sqrt,
                   s.metric,
                   s.seed});
  }
  return out;
}

// ======================= Naive reference kernels =======================

// Helpers to write the NN result — either key+value or value-only.
template <typename DataT, typename IdxT>
__device__ void writeNNResult(raft::KeyValuePair<IdxT, DataT>* out,
                              IdxT idx,
                              IdxT min_idx,
                              DataT min_dist)
{
  out[idx].key   = min_idx;
  out[idx].value = min_dist;
}

template <typename DataT, typename IdxT>
__device__ void writeNNResult(DataT* out, IdxT idx, IdxT /*min_idx*/, DataT min_dist)
{
  out[idx] = min_dist;
}

template <typename OutT, typename DataT, typename IdxT>
RAFT_KERNEL naiveFusedL2NNKernel(OutT* min,
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
    // L2 expanded formula: ||x - y||² = ||x||² + ||y||² - 2 * <x, y>
    DataT dot_product = DataT(0);
    for (IdxT i = 0; i < k; ++i) {
      dot_product += x[midx * k + i] * y[nidx * k + i];
    }

    DataT acc = x_norm_sq + yn[nidx] - DataT(2) * dot_product;

    if (sqrt) { acc = raft::sqrt(acc); }

    if (acc < min_dist) {
      min_dist = acc;
      min_idx  = nidx;
    }
  }

  writeNNResult(min, midx, min_idx, min_dist);
}

template <typename OutT, typename DataT, typename IdxT>
RAFT_KERNEL naiveFusedCosineNNKernel(OutT* min,
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

  writeNNResult(min, midx, min_idx, min_dist);
}

template <typename OutT, typename DataT, typename IdxT>
void naiveFusedDistanceNN(OutT* min,
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
      naiveFusedL2NNKernel<OutT, DataT, IdxT>
        <<<nblks, TPB, 0, stream>>>(min, x, y, xn, yn, m, n, k, sqrt);
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      naiveFusedCosineNNKernel<OutT, DataT, IdxT>
        <<<nblks, TPB, 0, stream>>>(min, x, y, xn, yn, m, n, k, sqrt);
      break;
    default: FAIL() << "Unsupported metric for naiveFusedDistanceNN";
  }
  RAFT_CUDA_TRY(cudaGetLastError());
}

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

// ======================= Common base for all fixtures =======================

template <typename DataT, typename IdxT>
using TestWithParam = ::testing::TestWithParam<FusedDistanceNNInputs<DataT, IdxT>>;

template <typename DataT, typename IdxT>
class FusedDistanceNNTestBase : public TestWithParam<DataT, IdxT> {
 protected:
  FusedDistanceNNTestBase()
    : params(TestWithParam<DataT, IdxT>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      x(params.m * params.k, stream),
      y(params.n * params.k, stream),
      xn(params.m, stream),
      yn(params.n, stream),
      workspace(params.m, stream)
  {
  }

  void generateData()
  {
    raft::random::RngState rng(params.seed);
    raft::random::uniform(handle, rng, x.data(), params.m * params.k, DataT(-1.0), DataT(1.0));
    raft::random::uniform(handle, rng, y.data(), params.n * params.k, DataT(-1.0), DataT(1.0));
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

  raft::resources handle;
  cudaStream_t stream;

  FusedDistanceNNInputs<DataT, IdxT> params;
  rmm::device_uvector<DataT> x, y, xn, yn;
  rmm::device_uvector<int> workspace;
};

// ======================= FusedDistanceNNTest =======================

template <typename DataT, typename IdxT>
class FusedDistanceNNTest : public FusedDistanceNNTestBase<DataT, IdxT> {
 public:
  FusedDistanceNNTest()
    : FusedDistanceNNTestBase<DataT, IdxT>(),
      min_ref(this->params.m, this->stream),
      min_actual(this->params.m, this->stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    this->generateData();
    this->computeNorms();
    computeReference();
    runActual(true);
    raft::resource::sync_stream(this->handle, this->stream);
  }

  void verify()
  {
    ASSERT_TRUE(cuvs::devArrMatch(min_ref.data(),
                                  min_actual.data(),
                                  this->params.m,
                                  KVPCompare<DataT>(this->params.tolerance),
                                  this->stream));
  }

 protected:
  void computeReference()
  {
    naiveFusedDistanceNN(min_ref.data(),
                         this->x.data(),
                         this->y.data(),
                         this->xn.data(),
                         this->yn.data(),
                         this->params.m,
                         this->params.n,
                         this->params.k,
                         this->params.sqrt,
                         this->params.metric,
                         this->stream);
  }

  void runActual(bool initOutBuffer)
  {
    fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(min_actual.data(),
                                                                           this->x.data(),
                                                                           this->y.data(),
                                                                           this->xn.data(),
                                                                           this->yn.data(),
                                                                           this->params.m,
                                                                           this->params.n,
                                                                           this->params.k,
                                                                           this->workspace.data(),
                                                                           this->params.sqrt,
                                                                           initOutBuffer,
                                                                           true,
                                                                           this->params.metric,
                                                                           0.0f,
                                                                           this->stream);
  }

  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_ref, min_actual;
};

// ======================= FusedDistanceNNPreInitTest =======================

template <typename DataT, typename IdxT>
class FusedDistanceNNPreInitTest : public FusedDistanceNNTest<DataT, IdxT> {
 public:
  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    this->generateData();
    this->computeNorms();
    this->computeReference();

    // Pre-initialize the output buffer with max values (like kmeans does)
    raft::KeyValuePair<IdxT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    thrust::fill(thrust::device,
                 this->min_actual.data(),
                 this->min_actual.data() + this->params.m,
                 initial_value);

    this->runActual(false);
    raft::resource::sync_stream(this->handle, this->stream);
  }
};

// ======================= FusedDistanceNNMinOnlyTest =======================

template <typename DataT, typename IdxT>
class FusedDistanceNNMinOnlyTest : public FusedDistanceNNTestBase<DataT, IdxT> {
 public:
  FusedDistanceNNMinOnlyTest()
    : FusedDistanceNNTestBase<DataT, IdxT>(),
      min_ref(this->params.m, this->stream),
      min_actual(this->params.m, this->stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    this->generateData();
    this->computeNorms();

    IdxT m = this->params.m;
    IdxT n = this->params.n;
    IdxT k = this->params.k;

    naiveFusedDistanceNN(min_ref.data(),
                         this->x.data(),
                         this->y.data(),
                         this->xn.data(),
                         this->yn.data(),
                         m,
                         n,
                         k,
                         this->params.sqrt,
                         this->params.metric,
                         this->stream);

    fusedDistanceNNMinReduce<DataT, DataT, IdxT>(min_actual.data(),
                                                 this->x.data(),
                                                 this->y.data(),
                                                 this->xn.data(),
                                                 this->yn.data(),
                                                 m,
                                                 n,
                                                 k,
                                                 this->workspace.data(),
                                                 this->params.sqrt,
                                                 true,
                                                 true,
                                                 this->params.metric,
                                                 0.0f,
                                                 this->stream);

    raft::resource::sync_stream(this->handle, this->stream);
  }

  void verify()
  {
    ASSERT_TRUE(cuvs::devArrMatch(min_ref.data(),
                                  min_actual.data(),
                                  this->params.m,
                                  cuvs::CompareApprox<DataT>(this->params.tolerance),
                                  this->stream));
  }

 protected:
  rmm::device_uvector<DataT> min_ref, min_actual;
};

// ======================= FusedDistanceNNLargeTest =======================

static constexpr int kCheckRows = 1024;

template <typename DataT, typename IdxT>
class FusedDistanceNNLargeTest : public FusedDistanceNNTestBase<DataT, IdxT> {
 public:
  FusedDistanceNNLargeTest()
    : FusedDistanceNNTestBase<DataT, IdxT>(),
      min_actual(this->params.m, this->stream),
      min_ref_first(kCheckRows, this->stream),
      min_ref_last(kCheckRows, this->stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    raft::common::nvtx::range fun_scope(
      "test::%s/%s", testInfo->test_suite_name(), testInfo->name());

    IdxT m = this->params.m;
    IdxT n = this->params.n;
    IdxT k = this->params.k;

    this->generateData();
    this->computeNorms();

    fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(min_actual.data(),
                                                                           this->x.data(),
                                                                           this->y.data(),
                                                                           this->xn.data(),
                                                                           this->yn.data(),
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           this->workspace.data(),
                                                                           this->params.sqrt,
                                                                           true,
                                                                           true,
                                                                           this->params.metric,
                                                                           0.0f,
                                                                           this->stream);

    naiveFusedDistanceNN(min_ref_first.data(),
                         this->x.data(),
                         this->y.data(),
                         this->xn.data(),
                         this->yn.data(),
                         (IdxT)kCheckRows,
                         n,
                         k,
                         this->params.sqrt,
                         this->params.metric,
                         this->stream);

    IdxT last_offset = m - (IdxT)kCheckRows;
    naiveFusedDistanceNN(min_ref_last.data(),
                         this->x.data() + last_offset * k,
                         this->y.data(),
                         this->xn.data() + last_offset,
                         this->yn.data(),
                         (IdxT)kCheckRows,
                         n,
                         k,
                         this->params.sqrt,
                         this->params.metric,
                         this->stream);

    raft::resource::sync_stream(this->handle, this->stream);
  }

  void verifyFirstRows()
  {
    ASSERT_TRUE(cuvs::devArrMatch(min_ref_first.data(),
                                  min_actual.data(),
                                  kCheckRows,
                                  KVPCompare<DataT>(this->params.tolerance),
                                  this->stream));
  }

  void verifyLastRows()
  {
    IdxT last_offset = this->params.m - kCheckRows;
    ASSERT_TRUE(cuvs::devArrMatch(min_ref_last.data(),
                                  min_actual.data() + last_offset,
                                  kCheckRows,
                                  KVPCompare<DataT>(this->params.tolerance),
                                  this->stream));
  }

 protected:
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_actual;
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_ref_first, min_ref_last;
};

// ======================= Test registration macros =======================

// clang-format off
#define FUSED_NN_TEST(SuiteName, TestAlias, FixtureClass, DataT, IdxT, InputsVar) \
  using TestAlias = FixtureClass<DataT, IdxT>;                                    \
  TEST_P(TestAlias, Result) { this->verify(); }                                   \
  INSTANTIATE_TEST_CASE_P(SuiteName, TestAlias, ::testing::ValuesIn(InputsVar))

#define FUSED_NN_LARGE_TEST(SuiteName, TestAlias, DataT, IdxT, InputsVar)         \
  using TestAlias = FusedDistanceNNLargeTest<DataT, IdxT>;                        \
  TEST_P(TestAlias, FirstRows) { this->verifyFirstRows(); }                       \
  TEST_P(TestAlias, LastRows)  { this->verifyLastRows(); }                        \
  INSTANTIATE_TEST_CASE_P(SuiteName, TestAlias, ::testing::ValuesIn(InputsVar))
// clang-format on

// ======================= Shape tables =======================

// --- Basic shapes (float gets all, compact subset for double / int64) ---
const std::vector<ShapeConfig> basicShapes = {
  // Basic L2 tests
  {32, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 128, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {256, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  // L2 sqrt tests
  {64, 64, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  {128, 128, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  // Skinny matrices (k < 32)
  {64, 64, 8, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {128, 128, 16, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {256, 256, 24, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  // Asymmetric dimensions
  {128, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {64, 128, 32, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {256, 64, 48, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  // Larger tests
  {512, 512, 64, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
  {1024, 1024, 32, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
  {1024, 512, 64, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
  // Cosine distance tests
  {64, 64, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
  {128, 128, 64, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
  {256, 256, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
  // Alignment edge cases
  {33, 33, 33, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {65, 65, 17, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {129, 127, 63, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
};

const std::vector<ShapeConfig> basicShapesCompact = {
  {64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 128, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  {128, 128, 16, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {64, 64, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
};

const std::vector<ShapeConfig> basicShapesInt64 = {
  {64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 128, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {256, 256, 32, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
};

const std::vector<ShapeConfig> basicShapesCompactInt64 = {
  {64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 128, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 5678ULL},
  {64, 64, 32, false, cuvs::distance::DistanceType::CosineExpanded, 2468ULL},
};

// --- Kmeans shapes ---
const std::vector<ShapeConfig> kmeansShapes = {
  // Small cluster counts
  {1000, 8, 64, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {1000, 16, 64, false, cuvs::distance::DistanceType::L2Expanded, 2222ULL},
  {1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 3333ULL},
  // Medium cluster counts
  {2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 4444ULL},
  {2000, 128, 128, false, cuvs::distance::DistanceType::L2Expanded, 5555ULL},
  {4096, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 6666ULL},
  // Large cluster counts
  {4096, 512, 32, false, cuvs::distance::DistanceType::L2Expanded, 7777ULL},
  {8192, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 8888ULL},
  // High-dimensional embeddings
  {1000, 32, 256, false, cuvs::distance::DistanceType::L2Expanded, 9999ULL},
  {1000, 32, 512, false, cuvs::distance::DistanceType::L2Expanded, 1010ULL},
  {2000, 64, 384, false, cuvs::distance::DistanceType::L2Expanded, 1212ULL},
  // L2 sqrt
  {1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1313ULL},
  {2000, 64, 256, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1414ULL},
  // Cosine distance
  {1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 1515ULL},
  {2000, 64, 256, false, cuvs::distance::DistanceType::CosineExpanded, 1616ULL},
  // Skinny matrices (k < 32)
  {1000, 64, 16, false, cuvs::distance::DistanceType::L2Expanded, 1717ULL},
  {2000, 128, 24, false, cuvs::distance::DistanceType::L2Expanded, 1818ULL},
  // Edge cases: very few clusters
  {5000, 2, 128, false, cuvs::distance::DistanceType::L2Expanded, 1919ULL},
  {5000, 4, 128, false, cuvs::distance::DistanceType::L2Expanded, 2020ULL},
  // Edge cases: many clusters relative to samples
  {512, 256, 64, false, cuvs::distance::DistanceType::L2Expanded, 2121ULL},
  {1024, 512, 64, false, cuvs::distance::DistanceType::L2Expanded, 2323ULL},
};

const std::vector<ShapeConfig> kmeansShapesCompact = {
  {1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 4444ULL},
  {1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1313ULL},
  {1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 1515ULL},
};

// --- Min-only shapes ---
const std::vector<ShapeConfig> minOnlyShapes = {
  {64, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {256, 128, 32, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
  {128, 64, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {1000, 32, 128, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {2000, 64, 256, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
};

const std::vector<ShapeConfig> minOnlyShapesCompact = {
  {64, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {128, 64, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
};

// --- Pre-init shapes ---
const std::vector<ShapeConfig> preInitShapes = {
  {1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {2000, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
  {1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

const std::vector<ShapeConfig> preInitShapesCompact = {
  {1000, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {1000, 32, 128, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {1000, 32, 128, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

// --- Large shapes ---
const std::vector<ShapeConfig> largeShapes = {
  // 1M samples
  {1 << 20, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {1 << 20, 64, 128, false, cuvs::distance::DistanceType::L2Expanded, 5678ULL},
  {1 << 20, 32, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {1 << 20, 32, 64, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
  // 10M samples
  {10000000, 32, 32, false, cuvs::distance::DistanceType::L2Expanded, 1111ULL},
  {10000000, 64, 64, false, cuvs::distance::DistanceType::L2Expanded, 2222ULL},
  {10000000, 32, 32, true, cuvs::distance::DistanceType::L2SqrtExpanded, 3333ULL},
  {10000000, 32, 32, false, cuvs::distance::DistanceType::CosineExpanded, 4444ULL},
};

const std::vector<ShapeConfig> largeShapesCompact = {
  {1 << 20, 32, 64, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {1 << 20, 32, 64, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  {1 << 20, 32, 64, false, cuvs::distance::DistanceType::CosineExpanded, 3456ULL},
};

// clang-format: off
// ======================= FusedDistanceNNBatchTest =======================
// Tests N-batching correctness: when N exceeds the 4 GB buffer descriptor
// limit the kernel batches over N.  For each query row we plant an identical
// copy of x[row,:] at a strategic position in y, guaranteeing row's NN is at
// a known global column with distance ≈ 0.
//
// Why planting instead of the naive reference kernel:
//   The naive kernel (naiveFusedL2NNKernel) assigns one GPU thread per query
//   row, and each thread loops over all n*k elements serially.  At the scales
//   needed to trigger N-batching (n ≈ 1M, k = 1024) that is ~1 billion FMAs
//   per thread — far too slow to be practical in a test.
//
//   Planting sidesteps this entirely: we copy x[row,:] into y[col,:] so the
//   answer for that row is known a priori (key = col, distance ≈ 0).  Because
//   random 1024-dimensional vectors are nearly orthogonal, the planted exact
//   copy always wins by a wide margin (expected random L2 ≈ 1365 vs ≈ 0),
//   making the test deterministic without a reference kernel.
//
// Plant positions are chosen relative to the N-batch boundary (nb_max) to
// cover: first/last columns, batch-0/batch-1 boundaries, and batch interiors.
// Positions are deduplicated so the 1-column-remainder case (n-1 == nb_max)
// doesn't double-plant.
// clang-format: on
#ifdef CUVS_CK_ENABLED
template <typename DataT, typename IdxT>
class FusedDistanceNNBatchTest : public FusedDistanceNNTestBase<DataT, IdxT> {
 public:
  FusedDistanceNNBatchTest()
    : FusedDistanceNNTestBase<DataT, IdxT>(), min_actual(this->params.m, this->stream)
  {
  }

  void SetUp() override
  {
    auto testInfo = testing::UnitTest::GetInstance()->current_test_info();

    this->generateData();

    IdxT k = this->params.k;
    IdxT n = this->params.n;
    IdxT m = this->params.m;

    // Compute nb_max to identify batch boundaries.
    // Mirrors max_per_batch_fused_nn() in dispatch_ck.cuh:
    //   limit  = (4GB - 1) / (k * sizeof(float))
    //   nb_max = roundDown(limit, NPerBlock)     [NPerBlock = 64 for all configs]
    constexpr int64_t NPerBlock = 64;
    int64_t limit =
      static_cast<int64_t>(static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) /
                           (static_cast<uint64_t>(k) * sizeof(DataT)));
    int64_t nb_max_raw = std::max<int64_t>((limit / NPerBlock) * NPerBlock, NPerBlock);
    IdxT nb_max        = static_cast<IdxT>(std::min<int64_t>(nb_max_raw, n));

    bool has_second_batch = (nb_max < n);
    IdxT nb_rem           = has_second_batch ? static_cast<IdxT>(n - nb_max) : IdxT(0);

    // Build plant list: for each (row, col), copy x[row,:] → y[col,:].
    // This guarantees row's NN is at global column col with distance ≈ 0.
    // Positions are deduplicated (matters when n-1 == nb_max, i.e. 1-col remainder).
    plants_.clear();
    IdxT row = 0;

    auto is_used = [&](IdxT col) {
      for (auto& [r, c] : plants_) {
        if (c == col) return true;
      }
      return false;
    };

    auto try_plant = [&](IdxT col) {
      if (row < m && !is_used(col)) { plants_.push_back({row++, col}); }
    };

    try_plant(IdxT(0));  // first column (batch 0)
    try_plant(n - 1);    // last column (batch 1 if batching occurs)

    if (has_second_batch) {
      try_plant(static_cast<IdxT>(nb_max - 1));  // last column of batch 0 (boundary)
      try_plant(static_cast<IdxT>(nb_max));      // first column of batch 1 (boundary)
      // middle of batch 1 (only if remainder has > 2 columns)
      if (nb_rem > 2) try_plant(static_cast<IdxT>(nb_max + nb_rem / 2));
    }

    try_plant(static_cast<IdxT>(nb_max / 2));  // middle of batch 0

    // Execute plants: copy x[row,:] → y[col,:]
    for (auto& [r, c] : plants_) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(this->y.data() + static_cast<int64_t>(c) * k,
                                    this->x.data() + static_cast<int64_t>(r) * k,
                                    k * sizeof(DataT),
                                    cudaMemcpyDeviceToDevice,
                                    this->stream));
    }

    this->computeNorms();

    fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(min_actual.data(),
                                                                           this->x.data(),
                                                                           this->y.data(),
                                                                           this->xn.data(),
                                                                           this->yn.data(),
                                                                           m,
                                                                           n,
                                                                           k,
                                                                           this->workspace.data(),
                                                                           this->params.sqrt,
                                                                           true,
                                                                           true,
                                                                           this->params.metric,
                                                                           0.0f,
                                                                           this->stream);

    raft::resource::sync_stream(this->handle, this->stream);
  }

  void verify()
  {
    IdxT m = this->params.m;
    IdxT n = this->params.n;

    std::vector<raft::KeyValuePair<IdxT, DataT>> h_actual(m);
    RAFT_CUDA_TRY(cudaMemcpy(h_actual.data(),
                             min_actual.data(),
                             m * sizeof(raft::KeyValuePair<IdxT, DataT>),
                             cudaMemcpyDeviceToHost));

    // Verify each planted row: key must be the planted global column index,
    // and distance must be ≈ 0 (identical vectors).
    for (auto& [row, expected_col] : plants_) {
      ASSERT_EQ(h_actual[row].key, expected_col)
        << "Row " << row << ": expected global column " << expected_col << " but got "
        << h_actual[row].key;
      // Tolerance is generous because the GEMM dot-product (MFMA) and the
      // separately-computed norms (rowNorm) use different accumulation orders,
      // so ||x||^2 + ||y||^2 - 2*(x·y) doesn't cancel to exactly 0.
      ASSERT_NEAR(h_actual[row].value, DataT(0), DataT(0.1))
        << "Row " << row << ": distance to planted copy should be ~0";
    }

    // All rows: keys must be valid global column indices in [0, n).
    for (IdxT i = 0; i < m; ++i) {
      ASSERT_GE(h_actual[i].key, IdxT(0)) << "Row " << i << ": negative key";
      ASSERT_LT(h_actual[i].key, n) << "Row " << i << ": key " << h_actual[i].key << " >= n";
    }
  }

 protected:
  std::vector<std::pair<IdxT, IdxT>> plants_;  // (row, expected_col)
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_actual;
};
#endif  // CUVS_CK_ENABLED

// ======================= Shape tables =======================

// --- CK skinny shapes ---
#ifdef CUVS_CK_ENABLED
const std::vector<ShapeConfig> ckSkinnyShapes = {
  // Skinny-K optimization boundary tests (K < 32 threshold)
  {64, 64, 2, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 4, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 8, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 16, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 24, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 31, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 32, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  {64, 64, 33, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  // Both metrics with skinny K
  {64, 64, 8, false, cuvs::distance::DistanceType::CosineExpanded, 5678ULL},
  {64, 64, 16, true, cuvs::distance::DistanceType::L2SqrtExpanded, 9012ULL},
  // Mixed tile sizes with skinny K (typical k-means scenarios)
  {32, 128, 8, false, cuvs::distance::DistanceType::L2Expanded, 2468ULL},
  {128, 32, 16, false, cuvs::distance::DistanceType::L2Expanded, 3579ULL},
  {256, 64, 12, false, cuvs::distance::DistanceType::L2Expanded, 4680ULL},
  // Legacy vector alignment tests
  {64, 64, 1, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {64, 64, 3, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {64, 64, 5, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  {64, 64, 7, false, cuvs::distance::DistanceType::L2Expanded, 1357ULL},
  // Edge cases
  {1, 1, 1, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {1, 64, 4, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {64, 1, 8, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {1, 1024, 16, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  // Mixed scenarios
  {33, 65, 6, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  {96, 32, 12, false, cuvs::distance::DistanceType::L2Expanded, 3456ULL},
  // Large-M stress tests with skinny K
  {100000, 64, 2, false, cuvs::distance::DistanceType::L2Expanded, 7890ULL},
  {200000, 32, 4, false, cuvs::distance::DistanceType::L2Expanded, 8901ULL},
  {150000, 128, 8, false, cuvs::distance::DistanceType::L2Expanded, 9012ULL},
  {80000, 256, 16, false, cuvs::distance::DistanceType::L2Expanded, 1123ULL},
  {500000, 16, 12, false, cuvs::distance::DistanceType::L2Expanded, 2234ULL},
  // Large-M with cosine distance
  {120000, 100, 24, false, cuvs::distance::DistanceType::CosineExpanded, 3345ULL},
  {250000, 50, 16, false, cuvs::distance::DistanceType::CosineExpanded, 4456ULL},
};

// N-batching shapes: n * k * sizeof(float) > 4 GB triggers N-batching.
// For k=1024, nb_max = 1,048,512, so any n > nb_max creates 2+ N-batches.
//
// Corner cases tested:
//   - 64-column remainder:  m=6 plants at first/last cols, both batch boundaries, batch interiors
//   - 1-column remainder:   extreme edge case, single-column second batch (entire tile is padding)
//   - Cosine distance:      verifies n_offset works with cosine epilogue (distance = 1 - cos = 0)
//   - L2 sqrt distance:     verifies n_offset works with sqrt epilogue (sqrt(0) = 0)
const std::vector<ShapeConfig> ckNBatchShapes = {
  // 2 N-batches, 64-column remainder (m=6 for comprehensive boundary planting)
  {6, 1048576, 1024, false, cuvs::distance::DistanceType::L2Expanded, 4242ULL},
  // 2 N-batches, 1-column remainder (smallest possible second batch)
  {4, 1048513, 1024, false, cuvs::distance::DistanceType::L2Expanded, 4243ULL},
  // 2 N-batches with cosine distance
  {4, 1048576, 1024, false, cuvs::distance::DistanceType::CosineExpanded, 4244ULL},
  // 2 N-batches with L2 sqrt
  {4, 1048576, 1024, true, cuvs::distance::DistanceType::L2SqrtExpanded, 4245ULL},
};

const std::vector<ShapeConfig> ckLargeSkinnyShapes = {
  // Large-M stress tests around 5M with skinny K
  {4800000, 64, 2, false, cuvs::distance::DistanceType::L2Expanded, 5567ULL},
  {5000000, 32, 4, false, cuvs::distance::DistanceType::L2Expanded, 6678ULL},
  {5200000, 128, 8, false, cuvs::distance::DistanceType::L2Expanded, 7789ULL},
  {4900000, 256, 16, false, cuvs::distance::DistanceType::L2Expanded, 8890ULL},
  {5100000, 16, 12, false, cuvs::distance::DistanceType::L2Expanded, 9901ULL},
  {4700000, 48, 6, false, cuvs::distance::DistanceType::L2Expanded, 1012ULL},
  {5300000, 80, 24, false, cuvs::distance::DistanceType::L2Expanded, 1123ULL},
  {4600000, 96, 10, false, cuvs::distance::DistanceType::L2Expanded, 1234ULL},
  // Large-M around 5M with cosine distance
  {5000000, 64, 8, false, cuvs::distance::DistanceType::CosineExpanded, 1345ULL},
  {4800000, 32, 16, false, cuvs::distance::DistanceType::CosineExpanded, 1456ULL},
  {5200000, 96, 12, false, cuvs::distance::DistanceType::CosineExpanded, 1567ULL},
  // Large-M around 5M with L2 sqrt
  {5000000, 128, 4, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1678ULL},
  {4900000, 64, 20, true, cuvs::distance::DistanceType::L2SqrtExpanded, 1789ULL},
};
#endif  // CUVS_CK_ENABLED

// ======================= Generate typed inputs =======================

// --- Basic ---
const auto inputsF       = makeInputs<float, int>(0.01f, basicShapes);
const auto inputsD       = makeInputs<double, int>(0.001, basicShapesCompact);
const auto inputsF_Int64 = makeInputs<float, int64_t>(0.01f, basicShapesInt64);
const auto inputsD_Int64 = makeInputs<double, int64_t>(0.001, basicShapesCompactInt64);

// --- Kmeans ---
const auto inputsKmeansF       = makeInputs<float, int>(0.01f, kmeansShapes);
const auto inputsKmeansD       = makeInputs<double, int>(0.001, kmeansShapesCompact);
const auto inputsKmeansF_Int64 = makeInputs<float, int64_t>(0.01f, kmeansShapesCompact);
const auto inputsKmeansD_Int64 = makeInputs<double, int64_t>(0.001, kmeansShapesCompact);

// --- Min-only ---
const auto inputsMinOnlyF       = makeInputs<float, int>(0.01f, minOnlyShapes);
const auto inputsMinOnlyD       = makeInputs<double, int>(0.001, minOnlyShapesCompact);
const auto inputsMinOnlyF_Int64 = makeInputs<float, int64_t>(0.01f, minOnlyShapesCompact);
const auto inputsMinOnlyD_Int64 = makeInputs<double, int64_t>(0.001, minOnlyShapesCompact);

// --- Pre-init ---
const auto inputsPreInitF       = makeInputs<float, int>(0.01f, preInitShapes);
const auto inputsPreInitD       = makeInputs<double, int>(0.001, preInitShapesCompact);
const auto inputsPreInitF_Int64 = makeInputs<float, int64_t>(0.01f, preInitShapesCompact);
const auto inputsPreInitD_Int64 = makeInputs<double, int64_t>(0.001, preInitShapesCompact);

// --- Large ---
const auto inputsLargeF       = makeInputs<float, int>(0.01f, largeShapes);
const auto inputsLargeD       = makeInputs<double, int>(0.001, largeShapesCompact);
const auto inputsLargeF_Int64 = makeInputs<float, int64_t>(0.01f, largeShapesCompact);
const auto inputsLargeD_Int64 = makeInputs<double, int64_t>(0.001, largeShapesCompact);

// --- CK ---
#ifdef CUVS_CK_ENABLED
const auto inputsCkSkinny      = makeInputs<float, int>(0.01f, ckSkinnyShapes);
const auto inputsCkLargeSkinny = makeInputs<float, int>(0.01f, ckLargeSkinnyShapes);
const auto inputsCkNBatch      = makeInputs<float, int>(0.01f, ckNBatchShapes);
#endif

// ======================= Test registrations =======================

// --- Basic ---
FUSED_NN_TEST(FusedDistanceNNTests, FusedDistanceNNTestF, FusedDistanceNNTest, float, int, inputsF);
FUSED_NN_TEST(
  FusedDistanceNNTests, FusedDistanceNNTestD, FusedDistanceNNTest, double, int, inputsD);
FUSED_NN_TEST(FusedDistanceNNTests,
              FusedDistanceNNTestF_Int64,
              FusedDistanceNNTest,
              float,
              int64_t,
              inputsF_Int64);
FUSED_NN_TEST(FusedDistanceNNTests,
              FusedDistanceNNTestD_Int64,
              FusedDistanceNNTest,
              double,
              int64_t,
              inputsD_Int64);

// --- Kmeans ---
FUSED_NN_TEST(FusedDistanceNNKmeansTests,
              FusedDistanceNNKmeansTestF,
              FusedDistanceNNTest,
              float,
              int,
              inputsKmeansF);
FUSED_NN_TEST(FusedDistanceNNKmeansTests,
              FusedDistanceNNKmeansTestD,
              FusedDistanceNNTest,
              double,
              int,
              inputsKmeansD);
FUSED_NN_TEST(FusedDistanceNNKmeansTests,
              FusedDistanceNNKmeansTestF_Int64,
              FusedDistanceNNTest,
              float,
              int64_t,
              inputsKmeansF_Int64);
FUSED_NN_TEST(FusedDistanceNNKmeansTests,
              FusedDistanceNNKmeansTestD_Int64,
              FusedDistanceNNTest,
              double,
              int64_t,
              inputsKmeansD_Int64);

// --- Min-only ---
FUSED_NN_TEST(FusedDistanceNNMinOnlyTests,
              FusedDistanceNNMinOnlyTestF,
              FusedDistanceNNMinOnlyTest,
              float,
              int,
              inputsMinOnlyF);
FUSED_NN_TEST(FusedDistanceNNMinOnlyTests,
              FusedDistanceNNMinOnlyTestD,
              FusedDistanceNNMinOnlyTest,
              double,
              int,
              inputsMinOnlyD);
FUSED_NN_TEST(FusedDistanceNNMinOnlyTests,
              FusedDistanceNNMinOnlyTestF_Int64,
              FusedDistanceNNMinOnlyTest,
              float,
              int64_t,
              inputsMinOnlyF_Int64);
FUSED_NN_TEST(FusedDistanceNNMinOnlyTests,
              FusedDistanceNNMinOnlyTestD_Int64,
              FusedDistanceNNMinOnlyTest,
              double,
              int64_t,
              inputsMinOnlyD_Int64);

// --- Pre-init ---
FUSED_NN_TEST(FusedDistanceNNPreInitTests,
              FusedDistanceNNPreInitTestF,
              FusedDistanceNNPreInitTest,
              float,
              int,
              inputsPreInitF);
FUSED_NN_TEST(FusedDistanceNNPreInitTests,
              FusedDistanceNNPreInitTestD,
              FusedDistanceNNPreInitTest,
              double,
              int,
              inputsPreInitD);
FUSED_NN_TEST(FusedDistanceNNPreInitTests,
              FusedDistanceNNPreInitTestF_Int64,
              FusedDistanceNNPreInitTest,
              float,
              int64_t,
              inputsPreInitF_Int64);
FUSED_NN_TEST(FusedDistanceNNPreInitTests,
              FusedDistanceNNPreInitTestD_Int64,
              FusedDistanceNNPreInitTest,
              double,
              int64_t,
              inputsPreInitD_Int64);

// --- Large ---
FUSED_NN_LARGE_TEST(FusedDistanceNNLargeTests, FusedDistanceNNLargeTestF, float, int, inputsLargeF);
FUSED_NN_LARGE_TEST(
  FusedDistanceNNLargeTests, FusedDistanceNNLargeTestD, double, int, inputsLargeD);
FUSED_NN_LARGE_TEST(
  FusedDistanceNNLargeTests, FusedDistanceNNLargeTestF_Int64, float, int64_t, inputsLargeF_Int64);
FUSED_NN_LARGE_TEST(
  FusedDistanceNNLargeTests, FusedDistanceNNLargeTestD_Int64, double, int64_t, inputsLargeD_Int64);

// --- CK ---
#ifdef CUVS_CK_ENABLED
FUSED_NN_TEST(FusedDistanceNNTests,
              FusedDistanceNNCkSkinnyTest,
              FusedDistanceNNTest,
              float,
              int,
              inputsCkSkinny);
FUSED_NN_LARGE_TEST(FusedDistanceNNCkLargeSkinnyTests,
                    FusedDistanceNNCkLargeSkinnyTest,
                    float,
                    int,
                    inputsCkLargeSkinny);
FUSED_NN_TEST(FusedDistanceNNNBatchTests,
              FusedDistanceNNNBatchTestF,
              FusedDistanceNNBatchTest,
              float,
              int,
              inputsCkNBatch);
#endif  // CUVS_CK_ENABLED

}  // namespace distance
}  // namespace cuvs
