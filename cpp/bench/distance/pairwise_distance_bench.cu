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

#include <cuvs/cuda_runtime.h>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kWarmupIterations = 3;

struct float_to_half_op {
  __host__ __device__ half operator()(float x) const
  {
#ifdef __HIP_PLATFORM_AMD__
    return __float2half(x);
#else
    return __float2half_rn(x);
#endif
  }
};

template <typename T>
void fill_random_matrices(raft::resources const& handle,
                          raft::random::RngState& r,
                          raft::device_matrix_view<T, int64_t, raft::layout_c_contiguous> x,
                          raft::device_matrix_view<T, int64_t, raft::layout_c_contiguous> y,
                          double lo,
                          double hi)
{
  if constexpr (std::is_same_v<T, half>) {
    auto x_float = raft::make_device_matrix<float>(handle, x.extent(0), x.extent(1));
    auto y_float = raft::make_device_matrix<float>(handle, y.extent(0), y.extent(1));
    raft::random::uniform(
      handle, r, x_float.data_handle(), x.size(), static_cast<float>(lo), static_cast<float>(hi));
    raft::random::uniform(
      handle, r, y_float.data_handle(), y.size(), static_cast<float>(lo), static_cast<float>(hi));
    raft::linalg::unary_op(handle, raft::make_const_mdspan(x_float.view()), x, float_to_half_op{});
    raft::linalg::unary_op(handle, raft::make_const_mdspan(y_float.view()), y, float_to_half_op{});
  } else {
    raft::random::uniform(
      handle, r, x.data_handle(), x.size(), static_cast<T>(lo), static_cast<T>(hi));
    raft::random::uniform(
      handle, r, y.data_handle(), y.size(), static_cast<T>(lo), static_cast<T>(hi));
  }
}

template <typename T>
void run_pairwise_distance_bench(
  benchmark::State& state, cuvs::distance::DistanceType metric, int64_t m, int64_t n, int64_t k)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  auto x    = raft::make_device_matrix<T>(handle, m, k);
  auto y    = raft::make_device_matrix<T>(handle, n, k);
  auto dist = raft::make_device_matrix<float>(handle, m, n);

  raft::random::RngState r(12345);
  fill_random_matrices(handle, r, x.view(), y.view(), -1.0, 1.0);
  raft::resource::sync_stream(handle, stream);

  float metric_arg = 2.0f;
  for (int i = 0; i < kWarmupIterations; ++i) {
    cuvs::distance::pairwise_distance(handle, x.view(), y.view(), dist.view(), metric, metric_arg);
    raft::resource::sync_stream(handle, stream);
  }

  cudaEvent_t start_ev = nullptr;
  cudaEvent_t stop_ev  = nullptr;
  RAFT_CUDA_TRY(cudaEventCreate(&start_ev));
  RAFT_CUDA_TRY(cudaEventCreate(&stop_ev));

  double gpu_time_total_s = 0.0;
  for (auto _ : state) {
    RAFT_CUDA_TRY(cudaEventRecord(start_ev, stream));
    cuvs::distance::pairwise_distance(handle, x.view(), y.view(), dist.view(), metric, metric_arg);
    RAFT_CUDA_TRY(cudaEventRecord(stop_ev, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(stop_ev));
    float ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start_ev, stop_ev));
    gpu_time_total_s += ms / 1000.0;
  }

  RAFT_CUDA_TRY(cudaEventDestroy(start_ev));
  RAFT_CUDA_TRY(cudaEventDestroy(stop_ev));

  state.counters["m"] = benchmark::Counter(static_cast<double>(m));
  state.counters["n"] = benchmark::Counter(static_cast<double>(n));
  state.counters["k"] = benchmark::Counter(static_cast<double>(k));
  state.counters["GPU time"] =
    benchmark::Counter(gpu_time_total_s, benchmark::Counter::kAvgIterations);
}

const auto kBenchArgs = std::vector<std::vector<int64_t>>{
  {1024, 4096, 8192, 16384}, {1024, 4096, 8192, 16384}, {16, 32, 64, 128, 256, 512, 1024}};

void register_benchmarks(const std::string& name, cuvs::distance::DistanceType metric)
{
  benchmark::RegisterBenchmark("pairwise_distance/" + name + "_float",
                               [metric](benchmark::State& state) {
                                 auto m = static_cast<int64_t>(state.range(0));
                                 auto n = static_cast<int64_t>(state.range(1));
                                 auto k = static_cast<int64_t>(state.range(2));
                                 run_pairwise_distance_bench<float>(state, metric, m, n, k);
                               })
    ->ArgsProduct({kBenchArgs[0], kBenchArgs[1], kBenchArgs[2]})
    ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("pairwise_distance/" + name + "_half",
                               [metric](benchmark::State& state) {
                                 auto m = static_cast<int64_t>(state.range(0));
                                 auto n = static_cast<int64_t>(state.range(1));
                                 auto k = static_cast<int64_t>(state.range(2));
                                 run_pairwise_distance_bench<half>(state, metric, m, n, k);
                               })
    ->ArgsProduct({kBenchArgs[0], kBenchArgs[1], kBenchArgs[2]})
    ->Unit(benchmark::kMillisecond);
}

}  // namespace

int main(int argc, char** argv)
{
  register_benchmarks("L2Expanded", cuvs::distance::DistanceType::L2Expanded);
  register_benchmarks("L2SqrtExpanded", cuvs::distance::DistanceType::L2SqrtExpanded);
  register_benchmarks("CosineExpanded", cuvs::distance::DistanceType::CosineExpanded);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
