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

// Parameterized gtest for CK fused-GEMM distance kernels vs. reference implementation.
// Templated on (DataT, DistT) to exercise different precision combinations.
// Supports CosineDistance and L2Expanded metrics.

#include "ck_fused_gemm.cuh"
#include "reference.cuh"

#include <raft/util/cuda_rt_essentials.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------
template <typename T>
constexpr const char* type_name();
template <>
constexpr const char* type_name<float>()
{
  return "f32";
}
template <>
constexpr const char* type_name<ck_tile::half_t>()
{
  return "f16";
}
template <>
constexpr const char* type_name<ck_tile::bf16_t>()
{
  return "bf16";
}

template <typename LayoutT>
constexpr const char* layout_name();
template <>
constexpr const char* layout_name<raft::layout_c_contiguous>()
{
  return "rm";
}
template <>
constexpr const char* layout_name<raft::layout_f_contiguous>()
{
  return "cm";
}

template <typename DataT_, typename DistT_, typename LayoutT_ = raft::layout_c_contiguous>
struct TypeCombo {
  using DataT   = DataT_;
  using DistT   = DistT_;
  using LayoutT = LayoutT_;
};

// ---------------------------------------------------------------------------
// Test parameters (type-independent)
// ---------------------------------------------------------------------------
struct CkDistanceTestParams {
  int64_t n_x;
  int64_t n_y;
  int64_t k;
  double tolerance;  // mean absolute error tolerance per element
};

std::ostream& operator<<(std::ostream& os, const CkDistanceTestParams& p)
{
  return os << "nx" << p.n_x << "_ny" << p.n_y << "_k" << p.k;
}

// ---------------------------------------------------------------------------
// Common test fixture base -- SetUp/TearDown for HIP timing events
// ---------------------------------------------------------------------------
template <typename Combo>
class CkDistanceTestBase : public ::testing::TestWithParam<CkDistanceTestParams> {
 protected:
  using DataT   = typename Combo::DataT;
  using DistT   = typename Combo::DistT;
  using LayoutT = typename Combo::LayoutT;

  void SetUp() override
  {
    RAFT_CUDA_TRY(hipEventCreate(&start_ev_));
    RAFT_CUDA_TRY(hipEventCreate(&stop_ev_));
    // CK Tile float warp GEMM (e.g. F32 16x16x4) is only implemented for __gfx9__; on other
    // arches it returns zero, so fp32 distance tests would fail. Skip when not on GFX9.
    int device = 0;
    RAFT_CUDA_TRY(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    RAFT_CUDA_TRY(hipGetDeviceProperties(&prop, device));
    if (std::strncmp(prop.gcnArchName, "gfx9", 4) != 0) {
      GTEST_SKIP() << "CK distance tests require GFX9 (CK float MFMA implemented only for gfx9); "
                   << "current arch: " << prop.gcnArchName;
    }
  }

  void TearDown() override
  {
    RAFT_CUDA_TRY(hipEventDestroy(start_ev_));
    RAFT_CUDA_TRY(hipEventDestroy(stop_ev_));
  }

  // Prepare random data in float (row-major, for reference) + convert to DataT in LayoutT.
  // Returns {x_d (LayoutT), y_d (LayoutT), x_f32 (row-major), y_f32 (row-major)}.
  auto PrepareData(raft::resources const& handle, int64_t n_x, int64_t n_y, int64_t k)
  {
    auto stream = raft::resource::get_cuda_stream(handle);

    // Row-major float data -- used as-is for the reference implementation.
    auto x_f32 = raft::make_device_matrix<float, int64_t>(handle, n_x, k);
    auto y_f32 = raft::make_device_matrix<float, int64_t>(handle, n_y, k);
    raft::random::RngState r(1234ULL);
    raft::random::uniform(handle, r, x_f32.data_handle(), n_x * k, -1.0f, 1.0f);
    raft::random::uniform(handle, r, y_f32.data_handle(), n_y * k, -1.0f, 1.0f);

    // DataT matrices in the target layout for the CK kernel.
    auto x_d = raft::make_device_matrix<DataT, int64_t, LayoutT>(handle, n_x, k);
    auto y_d = raft::make_device_matrix<DataT, int64_t, LayoutT>(handle, n_y, k);

    if constexpr (std::is_same_v<LayoutT, raft::layout_c_contiguous>) {
      // Row-major: flat element-wise conversion preserves layout.
      thrust::transform(raft::resource::get_thrust_policy(handle),
                        x_f32.data_handle(),
                        x_f32.data_handle() + n_x * k,
                        x_d.data_handle(),
                        [] __device__(float v) { return static_cast<DataT>(v); });
      thrust::transform(raft::resource::get_thrust_policy(handle),
                        y_f32.data_handle(),
                        y_f32.data_handle() + n_y * k,
                        y_d.data_handle(),
                        [] __device__(float v) { return static_cast<DataT>(v); });
    } else {
      // Col-major: transpose row-major float into col-major float, then convert to DataT.
      auto x_f32_cm =
        raft::make_device_matrix<float, int64_t, raft::layout_f_contiguous>(handle, n_x, k);
      auto y_f32_cm =
        raft::make_device_matrix<float, int64_t, raft::layout_f_contiguous>(handle, n_y, k);

      // Explicit transpose via map_offset:
      // Col-major flat index idx -> (row, col) = (idx % nrows, idx / nrows).
      // Read from row-major source at src[row * ncols + col].
      auto x_cm_vec =
        raft::make_device_vector_view<float, int64_t>(x_f32_cm.data_handle(), n_x * k);
      raft::linalg::map_offset(
        handle, x_cm_vec, [src = x_f32.data_handle(), n_x, k] __device__(size_t idx) {
          int64_t row = static_cast<int64_t>(idx) % n_x;
          int64_t col = static_cast<int64_t>(idx) / n_x;
          return src[row * k + col];
        });
      auto y_cm_vec =
        raft::make_device_vector_view<float, int64_t>(y_f32_cm.data_handle(), n_y * k);
      raft::linalg::map_offset(
        handle, y_cm_vec, [src = y_f32.data_handle(), n_y, k] __device__(size_t idx) {
          int64_t row = static_cast<int64_t>(idx) % n_y;
          int64_t col = static_cast<int64_t>(idx) / n_y;
          return src[row * k + col];
        });

      // Convert col-major float -> col-major DataT (flat element-wise, same physical layout).
      thrust::transform(raft::resource::get_thrust_policy(handle),
                        x_f32_cm.data_handle(),
                        x_f32_cm.data_handle() + n_x * k,
                        x_d.data_handle(),
                        [] __device__(float v) { return static_cast<DataT>(v); });
      thrust::transform(raft::resource::get_thrust_policy(handle),
                        y_f32_cm.data_handle(),
                        y_f32_cm.data_handle() + n_y * k,
                        y_d.data_handle(),
                        [] __device__(float v) { return static_cast<DataT>(v); });
    }

    return std::make_tuple(std::move(x_d), std::move(y_d), std::move(x_f32), std::move(y_f32));
  }

  // Compare CK output (DistT) against reference output (float).
  void CompareResults(const char* metric_name,
                      raft::resources const& handle,
                      int64_t n_x,
                      int64_t n_y,
                      int64_t k,
                      float ms_ck,
                      float ms_ref,
                      raft::device_matrix<DistT, int64_t> const& distance_ck,
                      raft::device_matrix<float, int64_t> const& distance_ref,
                      double tolerance)
  {
    auto stream = raft::resource::get_cuda_stream(handle);

    std::cout << std::fixed << std::setprecision(3) << "  [" << metric_name << " "
              << type_name<DataT>() << "/" << type_name<DistT>() << " " << layout_name<LayoutT>()
              << " " << n_x << "x" << n_y << " k=" << k << "] reference: " << ms_ref
              << " ms, ck: " << ms_ck << " ms\n";

    auto ref_host = raft::make_host_matrix<float, int64_t>(n_x, n_y);
    auto ck_host  = raft::make_host_matrix<DistT, int64_t>(n_x, n_y);
    raft::copy(ref_host.data_handle(), distance_ref.data_handle(), n_x * n_y, stream);
    raft::copy(ck_host.data_handle(), distance_ck.data_handle(), n_x * n_y, stream);
    raft::resource::sync_stream(handle);

    double sum_abs_err  = 0.0;
    double max_abs_err  = 0.0;
    int64_t max_err_idx = 0;
    for (int64_t i = 0; i < n_x * n_y; ++i) {
      double err = std::abs(static_cast<double>(ref_host.data_handle()[i]) -
                            static_cast<double>(ck_host.data_handle()[i]));
      sum_abs_err += err;
      if (err > max_abs_err) {
        max_abs_err = err;
        max_err_idx = i;
      }
    }
    double mean_abs_err = sum_abs_err / static_cast<double>(n_x * n_y);

    std::cout << std::scientific << std::setprecision(4) << "  mean_abs_err=" << mean_abs_err
              << "  max_abs_err=" << max_abs_err << "\n";

    EXPECT_LE(mean_abs_err, tolerance)
      << "Mean absolute error (" << mean_abs_err << ") exceeds tolerance (" << tolerance << ")";

    double vis_threshold = GetVisThreshold();
    if (max_abs_err > vis_threshold) {
      VisualizeError(
        n_x, n_y, ref_host.data_handle(), ck_host.data_handle(), max_err_idx, max_abs_err);
    }
  }

  // Read threshold once from CK_VIS_THRESHOLD env var (default: +inf, i.e. never visualize).
  static double GetVisThreshold()
  {
    static const double val = []() {
      const char* env = std::getenv("CK_VIS_THRESHOLD");
      return env ? std::stod(env) : std::numeric_limits<double>::infinity();
    }();
    return val;
  }

  // Print a patch around the max-error element and an ASCII error heatmap.
  void VisualizeError(int64_t n_x,
                      int64_t n_y,
                      const float* ref,
                      const DistT* ck,
                      int64_t max_err_idx,
                      double max_abs_err)
  {
    int64_t max_r = max_err_idx / n_y;
    int64_t max_c = max_err_idx % n_y;

    std::cout << "\n  --- Error visualization (CK_VIS_THRESHOLD=" << std::scientific
              << std::setprecision(1) << GetVisThreshold() << ") ---\n"
              << std::fixed << std::setprecision(6) << "  Max error at (" << max_r << ", " << max_c
              << "): ref=" << ref[max_err_idx] << "  ck=" << static_cast<double>(ck[max_err_idx])
              << "  |err|=" << std::scientific << max_abs_err << "\n";

    // 7x7 patch centred on the maximum error
    constexpr int64_t R = 3;
    int64_t r0          = std::max<int64_t>(0, max_r - R);
    int64_t r1          = std::min(n_x, max_r + R + 1);
    int64_t c0          = std::max<int64_t>(0, max_c - R);
    int64_t c1          = std::min(n_y, max_c + R + 1);

    auto print_patch = [&](const char* label, auto val_fn) {
      std::cout << "\n  " << label << " [" << r0 << ":" << r1 << ", " << c0 << ":" << c1 << "]:\n";
      for (int64_t r = r0; r < r1; ++r) {
        std::cout << "    ";
        for (int64_t c = c0; c < c1; ++c) {
          bool at_max = (r == max_r && c == max_c);
          std::cout << (at_max ? "[" : " ") << std::fixed << std::setprecision(4) << std::setw(9)
                    << val_fn(r * n_y + c) << (at_max ? "]" : " ");
        }
        std::cout << "\n";
      }
    };

    print_patch("Reference", [&](int64_t i) { return static_cast<double>(ref[i]); });
    print_patch("CK kernel", [&](int64_t i) { return static_cast<double>(ck[i]); });
    print_patch("|Error|  ", [&](int64_t i) {
      return std::abs(static_cast<double>(ref[i]) - static_cast<double>(ck[i]));
    });

    // Down-sampled ASCII error heatmap (max error in each cell)
    constexpr int64_t kMaxRows = 40;
    constexpr int64_t kMaxCols = 80;
    int64_t hm_r               = std::min(n_x, kMaxRows);
    int64_t hm_c               = std::min(n_y, kMaxCols);

    std::cout << "\n  Error heatmap (" << n_x << "x" << n_y << " -> " << hm_r << "x" << hm_c
              << ",  max in each cell):\n"
              << "  ' '=0  .=<1e-6  :=<1e-4  +=<1e-2  *=<1e-1  #>=1e-1\n";

    for (int64_t i = 0; i < hm_r; ++i) {
      int64_t rs = i * n_x / hm_r, re = (i + 1) * n_x / hm_r;
      std::cout << "  |";
      for (int64_t j = 0; j < hm_c; ++j) {
        int64_t cs = j * n_y / hm_c, ce = (j + 1) * n_y / hm_c;
        double mx = 0.0;
        for (int64_t r = rs; r < re; ++r)
          for (int64_t c = cs; c < ce; ++c)
            mx = std::max(mx,
                          std::abs(static_cast<double>(ref[r * n_y + c]) -
                                   static_cast<double>(ck[r * n_y + c])));
        // clang-format off
        char ch = (mx == 0.0) ? ' '
                : (mx < 1e-6) ? '.'
                : (mx < 1e-4) ? ':'
                : (mx < 1e-2) ? '+'
                : (mx < 1e-1) ? '*'
                :               '#';
        // clang-format on
        std::cout << ch;
      }
      std::cout << "|\n";
    }

    std::cout << "  --- End visualization ---\n\n";
  }

  hipEvent_t start_ev_ = nullptr;
  hipEvent_t stop_ev_  = nullptr;
};

// ---------------------------------------------------------------------------
// Cosine distance fixture
// ---------------------------------------------------------------------------
template <typename Combo>
class CkCosineDistanceTestBase : public CkDistanceTestBase<Combo> {
 public:
  // f16/bf16 inputs incur quantization error on top of the f32 baseline tolerance.
  static constexpr double toleranceScale()
  {
    using DataT = typename Combo::DataT;
    if constexpr (std::is_same_v<DataT, float>)
      return 1.0;
    else
      return 5.0;  // half_t and bf16_t
  }

  void RunTest()
  {
    using DataT                   = typename Combo::DataT;
    using DistT                   = typename Combo::DistT;
    using LayoutT                 = typename Combo::LayoutT;
    auto [n_x, n_y, k, tolerance] = this->GetParam();
    tolerance *= toleranceScale();

    raft::resources handle;
    auto stream                   = raft::resource::get_cuda_stream(handle);
    auto [x_d, y_d, x_f32, y_f32] = this->PrepareData(handle, n_x, n_y, k);

    auto x_view =
      raft::make_device_matrix_view<const DataT, int64_t, LayoutT>(x_d.data_handle(), n_x, k);
    auto y_view =
      raft::make_device_matrix_view<const DataT, int64_t, LayoutT>(y_d.data_handle(), n_y, k);

    float ms_ck = 0.f;
    RAFT_CUDA_TRY(hipEventRecord(this->start_ev_, stream));
    auto distance_ck = compute_ck_cosine_distance<DataT, DistT, LayoutT>(handle, x_view, y_view);
    RAFT_CUDA_TRY(hipEventRecord(this->stop_ev_, stream));
    RAFT_CUDA_TRY(hipEventSynchronize(this->stop_ev_));
    RAFT_CUDA_TRY(hipEventElapsedTime(&ms_ck, this->start_ev_, this->stop_ev_));

    // Reference always uses row-major float inputs.
    auto x_f32_view =
      raft::make_device_matrix_view<const float, int64_t, raft::layout_c_contiguous>(
        x_f32.data_handle(), n_x, k);
    auto y_f32_view =
      raft::make_device_matrix_view<const float, int64_t, raft::layout_c_contiguous>(
        y_f32.data_handle(), n_y, k);

    float ms_ref = 0.f;
    RAFT_CUDA_TRY(hipEventRecord(this->start_ev_, stream));
    auto distance_ref = compute_reference_cosine_distance(handle, x_f32_view, y_f32_view);
    RAFT_CUDA_TRY(hipEventRecord(this->stop_ev_, stream));
    RAFT_CUDA_TRY(hipEventSynchronize(this->stop_ev_));
    RAFT_CUDA_TRY(hipEventElapsedTime(&ms_ref, this->start_ev_, this->stop_ev_));

    this->CompareResults(
      "cosine", handle, n_x, n_y, k, ms_ck, ms_ref, distance_ck, distance_ref, tolerance);
  }
};

// ---------------------------------------------------------------------------
// L2Expanded distance fixture
// ---------------------------------------------------------------------------
template <typename Combo>
class CkL2ExpandedTestBase : public CkDistanceTestBase<Combo> {
 public:
  // L2Expanded squaring amplifies low-precision rounding; bf16 (7-bit mantissa) is worst.
  // Scaled to cover K up to 512.
  static constexpr double toleranceScale()
  {
    using DataT = typename Combo::DataT;
    if constexpr (std::is_same_v<DataT, float>)
      return 1.0;
    else if constexpr (std::is_same_v<DataT, ck_tile::bf16_t>)
      return 80.0;
    else
      return 8.0;  // half_t
  }

  void RunTest()
  {
    using DataT                   = typename Combo::DataT;
    using DistT                   = typename Combo::DistT;
    using LayoutT                 = typename Combo::LayoutT;
    auto [n_x, n_y, k, tolerance] = this->GetParam();
    tolerance *= toleranceScale();

    raft::resources handle;
    auto stream                   = raft::resource::get_cuda_stream(handle);
    auto [x_d, y_d, x_f32, y_f32] = this->PrepareData(handle, n_x, n_y, k);

    auto x_view =
      raft::make_device_matrix_view<const DataT, int64_t, LayoutT>(x_d.data_handle(), n_x, k);
    auto y_view =
      raft::make_device_matrix_view<const DataT, int64_t, LayoutT>(y_d.data_handle(), n_y, k);

    float ms_ck = 0.f;
    RAFT_CUDA_TRY(hipEventRecord(this->start_ev_, stream));
    auto distance_ck =
      compute_ck_l2_expanded_distance<DataT, DistT, LayoutT>(handle, x_view, y_view);
    RAFT_CUDA_TRY(hipEventRecord(this->stop_ev_, stream));
    RAFT_CUDA_TRY(hipEventSynchronize(this->stop_ev_));
    RAFT_CUDA_TRY(hipEventElapsedTime(&ms_ck, this->start_ev_, this->stop_ev_));

    // Reference always uses row-major float inputs.
    auto x_f32_view =
      raft::make_device_matrix_view<const float, int64_t, raft::layout_c_contiguous>(
        x_f32.data_handle(), n_x, k);
    auto y_f32_view =
      raft::make_device_matrix_view<const float, int64_t, raft::layout_c_contiguous>(
        y_f32.data_handle(), n_y, k);

    float ms_ref = 0.f;
    RAFT_CUDA_TRY(hipEventRecord(this->start_ev_, stream));
    auto distance_ref = compute_reference_l2_expanded_distance(handle, x_f32_view, y_f32_view);
    RAFT_CUDA_TRY(hipEventRecord(this->stop_ev_, stream));
    RAFT_CUDA_TRY(hipEventSynchronize(this->stop_ev_));
    RAFT_CUDA_TRY(hipEventElapsedTime(&ms_ref, this->start_ev_, this->stop_ev_));

    this->CompareResults(
      "L2Expanded", handle, n_x, n_y, k, ms_ck, ms_ref, distance_ck, distance_ref, tolerance);
  }
};

// ---------------------------------------------------------------------------
// Macro to stamp out a concrete fixture, TEST_P, and INSTANTIATE for any
// metric / type combination.
// ---------------------------------------------------------------------------
// clang-format off
#define CK_DISTANCE_TEST_SUITE(Metric, Name, FixtureBase, ComboType, ...)                   \
  using CkDistTest_##Metric##_##Name = FixtureBase<ComboType>;                              \
  TEST_P(CkDistTest_##Metric##_##Name, MatchesReference) { this->RunTest(); }               \
  INSTANTIATE_TEST_SUITE_P(                                                                  \
    Metric##_##Name, CkDistTest_##Metric##_##Name, __VA_ARGS__,                             \
    [](const ::testing::TestParamInfo<CkDistanceTestParams>& info) {                         \
      std::ostringstream oss;                                                                \
      oss << info.param;                                                                     \
      return oss.str();                                                                      \
    })
// clang-format on

// ---------------------------------------------------------------------------
// Type combo aliases -- row-major (default)
// ---------------------------------------------------------------------------
using Float32Combo  = TypeCombo<float, float>;
using Float16Combo  = TypeCombo<ck_tile::half_t, float>;
using BFloat16Combo = TypeCombo<ck_tile::bf16_t, float>;

// ---------------------------------------------------------------------------
// Type combo aliases -- col-major
// ---------------------------------------------------------------------------
using Float32ColCombo  = TypeCombo<float, float, raft::layout_f_contiguous>;
using Float16ColCombo  = TypeCombo<ck_tile::half_t, float, raft::layout_f_contiguous>;
using BFloat16ColCombo = TypeCombo<ck_tile::bf16_t, float, raft::layout_f_contiguous>;

// ---------------------------------------------------------------------------
// Test parameters -- grouped by category, sorted by increasing size.
// ---------------------------------------------------------------------------

// Aligned power-of-two dimensions -- basic GEMM + epilogue correctness.
static constexpr auto kAlignedParams = {
  CkDistanceTestParams{.n_x = 64, .n_y = 64, .k = 4, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 64, .n_y = 64, .k = 8, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 64, .n_y = 64, .k = 32, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 64, .n_y = 64, .k = 128, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 128, .n_y = 128, .k = 32, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 128, .n_y = 128, .k = 64, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 128, .n_y = 128, .k = 128, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 256, .n_y = 256, .k = 64, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 256, .n_y = 256, .k = 128, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 512, .n_y = 512, .k = 128, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 1024, .n_y = 1024, .k = 256, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 1024, .n_y = 1024, .k = 512, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 4096, .n_y = 4096, .k = 128, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 8192, .n_y = 512, .k = 1024, .tolerance = 1e-2},
};

// Large N that pushes M*N*sizeof(float) past the 4 GB V# descriptor limit
// at M >= 128.  Exercises the M-batching workaround in ck_fused_gemm.cuh.
// N = (65536 + 128) * 128 = 8,404,992.
static constexpr auto kLargeNParams = {
  CkDistanceTestParams{.n_x = 128, .n_y = (65536 + 128) * 128, .k = 4, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 128, .n_y = (65536 + 128) * 128, .k = 8, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 128, .n_y = (65536 + 128) * 128, .k = 32, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 128, .n_y = (65536 + 128) * 128, .k = 128, .tolerance = 1e-3},
};

// Non-aligned dimensions -- row-major only (exercises kPadM / scalar-vec safe kernel).
static constexpr auto kNonAlignedParams = {
  // Non-aligned M only (kPadM; fast vectorized kernel)
  CkDistanceTestParams{.n_x = 1025, .n_y = 4096, .k = 128, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 10001, .n_y = 4096, .k = 128, .tolerance = 1e-3},
  // Non-aligned N only (scalar-vector safe kernel fallback)
  CkDistanceTestParams{.n_x = 128, .n_y = 127, .k = 64, .tolerance = 1e-3},
  // Non-aligned K only
  CkDistanceTestParams{.n_x = 256, .n_y = 256, .k = 127, .tolerance = 1e-3},
  // Non-aligned M, N, and K simultaneously
  CkDistanceTestParams{.n_x = 1025, .n_y = 1023, .k = 127, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 63, .n_y = 63, .k = 127, .tolerance = 1e-3},
  CkDistanceTestParams{.n_x = 31, .n_y = 31, .k = 31, .tolerance = 1e-3},
};

// Row-major: all parameter categories.
static const std::vector<CkDistanceTestParams> kRowMajorTestParams = []() {
  std::vector<CkDistanceTestParams> params;
  params.insert(params.end(), kAlignedParams.begin(), kAlignedParams.end());
  params.insert(params.end(), kLargeNParams.begin(), kLargeNParams.end());
  params.insert(params.end(), kNonAlignedParams.begin(), kNonAlignedParams.end());
  return params;
}();

// Col-major: only aligned dims (scalar-vec safe kernel is incompatible with
// the pipeline's A-shuffle transpose; M, N, K must satisfy vector alignment).
static const std::vector<CkDistanceTestParams> kColumnMajorTestParams = []() {
  std::vector<CkDistanceTestParams> params;
  params.insert(params.end(), kAlignedParams.begin(), kAlignedParams.end());
  return params;
}();

// ===========================================================================
//  Cosine distance test suites
// ===========================================================================
// clang-format off
CK_DISTANCE_TEST_SUITE(Cosine, f32, CkCosineDistanceTestBase, Float32Combo,
  ::testing::ValuesIn(kRowMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(Cosine, f16, CkCosineDistanceTestBase, Float16Combo,
  ::testing::ValuesIn(kRowMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(Cosine, bf16, CkCosineDistanceTestBase, BFloat16Combo,
  ::testing::ValuesIn(kRowMajorTestParams)
);

// ===========================================================================
//  L2Expanded distance test suites
// ===========================================================================
CK_DISTANCE_TEST_SUITE(L2Expanded, f32, CkL2ExpandedTestBase, Float32Combo,
  ::testing::ValuesIn(kRowMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(L2Expanded, f16, CkL2ExpandedTestBase, Float16Combo,
  ::testing::ValuesIn(kRowMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(L2Expanded, bf16, CkL2ExpandedTestBase, BFloat16Combo,
  ::testing::ValuesIn(kRowMajorTestParams)
);

// ===========================================================================
//  Column-major Cosine distance test suites
// ===========================================================================
// Col-major: only the vectorized kernel is available (see kColumnMajorTestParams).
CK_DISTANCE_TEST_SUITE(Cosine_cm, f32, CkCosineDistanceTestBase, Float32ColCombo,
  ::testing::ValuesIn(kColumnMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(Cosine_cm, f16, CkCosineDistanceTestBase, Float16ColCombo,
  ::testing::ValuesIn(kColumnMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(Cosine_cm, bf16, CkCosineDistanceTestBase, BFloat16ColCombo,
  ::testing::ValuesIn(kColumnMajorTestParams)
);

// ===========================================================================
//  Column-major L2Expanded distance test suites
// ===========================================================================
CK_DISTANCE_TEST_SUITE(L2Expanded_cm, f32, CkL2ExpandedTestBase, Float32ColCombo,
  ::testing::ValuesIn(kColumnMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(L2Expanded_cm, f16, CkL2ExpandedTestBase, Float16ColCombo,
  ::testing::ValuesIn(kColumnMajorTestParams)
);

CK_DISTANCE_TEST_SUITE(L2Expanded_cm, bf16, CkL2ExpandedTestBase, BFloat16ColCombo,
  ::testing::ValuesIn(kColumnMajorTestParams)
);
// clang-format on

// ===========================================================================
//  Column-major: verify non-aligned dimensions throw std::runtime_error.
//  The scalar-vector safe kernel is incompatible with column-major (the
//  pipeline's A-shuffle transpose requires matching tile distributions that
//  break with VectorSizeA=1), so non-aligned dims must be rejected.
// ===========================================================================
TEST(CkDistColMajorReject, CosineThrowsOnNonAlignedM)
{
  raft::resources handle;
  constexpr int64_t m = 65, n = 64, k = 128;
  auto x      = raft::make_device_matrix<float, int64_t>(handle, m, k);
  auto y      = raft::make_device_matrix<float, int64_t>(handle, n, k);
  auto x_view = raft::make_device_matrix_view<const float, int64_t, raft::layout_f_contiguous>(
    x.data_handle(), m, k);
  auto y_view = raft::make_device_matrix_view<const float, int64_t, raft::layout_f_contiguous>(
    y.data_handle(), n, k);
  EXPECT_THROW(
    (compute_ck_cosine_distance<float, float, raft::layout_f_contiguous>(handle, x_view, y_view)),
    std::runtime_error);
}

TEST(CkDistColMajorReject, L2ExpandedThrowsOnNonAlignedM)
{
  raft::resources handle;
  constexpr int64_t m = 65, n = 64, k = 128;
  auto x      = raft::make_device_matrix<float, int64_t>(handle, m, k);
  auto y      = raft::make_device_matrix<float, int64_t>(handle, n, k);
  auto x_view = raft::make_device_matrix_view<const float, int64_t, raft::layout_f_contiguous>(
    x.data_handle(), m, k);
  auto y_view = raft::make_device_matrix_view<const float, int64_t, raft::layout_f_contiguous>(
    y.data_handle(), n, k);
  EXPECT_THROW((compute_ck_l2_expanded_distance<float, float, raft::layout_f_contiguous>(
                 handle, x_view, y_view)),
               std::runtime_error);
}
