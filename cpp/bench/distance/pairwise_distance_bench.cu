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

template <typename T>
auto make_random_matrices(raft::resources const& handle,
                          raft::random::RngState& r,
                          int64_t n_rows_x,
                          int64_t n_rows_y,
                          int64_t n_cols)
  -> std::pair<raft::device_matrix<T, int64_t, raft::layout_c_contiguous>,
               raft::device_matrix<T, int64_t, raft::layout_c_contiguous>>
{
  auto x_float = raft::make_device_matrix<float>(handle, n_rows_x, n_cols);
  auto y_float = raft::make_device_matrix<float>(handle, n_rows_y, n_cols);

  static constexpr auto kLowerBound = -1.0f;
  static constexpr auto kUpperBound = 1.0f;
  raft::random::uniform(handle, r, x_float.data_handle(), x_float.size(), kLowerBound, kUpperBound);
  raft::random::uniform(handle, r, y_float.data_handle(), y_float.size(), kLowerBound, kUpperBound);

  if constexpr (std::is_same_v<T, half>) {
    struct float_to_half_op {
      __host__ __device__ half operator()(float x) const { return __float2half(x); }
    };
    auto x = raft::make_device_matrix<T>(handle, n_rows_x, n_cols);
    auto y = raft::make_device_matrix<T>(handle, n_rows_y, n_cols);
    raft::linalg::unary_op(
      handle, raft::make_const_mdspan(x_float.view()), x.view(), float_to_half_op{});
    raft::linalg::unary_op(
      handle, raft::make_const_mdspan(y_float.view()), y.view(), float_to_half_op{});
    return {x, y};
  } else {
    return {x_float, y_float};
  }
}

template <typename T>
void run_pairwise_distance_bench(
  benchmark::State& state, cuvs::distance::DistanceType metric, int64_t m, int64_t n, int64_t k)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  raft::random::RngState r(12345);
  auto [x, y] = make_random_matrices<T>(handle, r, m, n, k);
  auto dist   = raft::make_device_matrix<float>(handle, m, n);
  raft::resource::sync_stream(handle, stream);

  for (int i = 0; i < kWarmupIterations; ++i) {
    cuvs::distance::pairwise_distance(handle, x.view(), y.view(), dist.view(), metric);
  }
  raft::resource::sync_stream(handle, stream);

  cudaEvent_t start_ev = nullptr;
  cudaEvent_t stop_ev  = nullptr;
  RAFT_CUDA_TRY(cudaEventCreate(&start_ev));
  RAFT_CUDA_TRY(cudaEventCreate(&stop_ev));

  double gpu_time_total_s = 0.0;
  for (auto _ : state) {
    RAFT_CUDA_TRY(cudaEventRecord(start_ev, stream));
    cuvs::distance::pairwise_distance(handle, x.view(), y.view(), dist.view(), metric);
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

template <typename T>
void register_benchmark_for_type(const std::string& name, cuvs::distance::DistanceType metric)
{
  auto type_name = typeid(T).name();
  benchmark::RegisterBenchmark("pairwise_distance/" + name + "_" + type_name,
                               [metric](benchmark::State& state) {
                                 auto m = static_cast<int64_t>(state.range(0));
                                 auto n = static_cast<int64_t>(state.range(1));
                                 auto k = static_cast<int64_t>(state.range(2));
                                 run_pairwise_distance_bench<T>(state, metric, m, n, k);
                               })
    ->ArgsProduct({kBenchArgs[0], kBenchArgs[1], kBenchArgs[2]})
    ->Unit(benchmark::kMillisecond);
}

void register_benchmarks(const std::string& name, cuvs::distance::DistanceType metric)
{
  register_benchmark_for_type<float>(name, metric);
  register_benchmark_for_type<half>(name, metric);
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
