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

#include <distance/fused_distance_nn-inl.cuh>

#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kWarmupIterations = 3;

template <typename DataT, typename IdxT>
void run_fused_distance_nn_bench(benchmark::State& state,
                                 cuvs::distance::DistanceType metric,
                                 int64_t m,
                                 int64_t n,
                                 int64_t k,
                                 bool sqrt)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  raft::random::RngState r(12345);

  rmm::device_uvector<DataT> x(m * k, stream);
  rmm::device_uvector<DataT> y(n * k, stream);
  rmm::device_uvector<DataT> xn(m, stream);
  rmm::device_uvector<DataT> yn(n, stream);
  rmm::device_uvector<raft::KeyValuePair<IdxT, DataT>> min_out(m, stream);
  rmm::device_uvector<int> workspace(m, stream);

  raft::random::uniform(handle, r, x.data(), m * k, DataT(-1.0), DataT(1.0));
  raft::random::uniform(handle, r, y.data(), n * k, DataT(-1.0), DataT(1.0));

  if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      xn.data(), x.data(), k, m, stream, raft::sqrt_op{});
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      yn.data(), y.data(), k, n, stream, raft::sqrt_op{});
  } else {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(xn.data(), x.data(), k, m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(yn.data(), y.data(), k, n, stream);
  }

  raft::resource::sync_stream(handle, stream);

  for (int i = 0; i < kWarmupIterations; ++i) {
    cuvs::distance::fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(
      min_out.data(),
      x.data(),
      y.data(),
      xn.data(),
      yn.data(),
      m,
      n,
      k,
      workspace.data(),
      sqrt,
      true,
      true,
      metric,
      0.0f,
      stream);
  }
  raft::resource::sync_stream(handle, stream);

  cudaEvent_t start_ev = nullptr;
  cudaEvent_t stop_ev  = nullptr;
  RAFT_CUDA_TRY(cudaEventCreate(&start_ev));
  RAFT_CUDA_TRY(cudaEventCreate(&stop_ev));

  double gpu_time_total_s = 0.0;
  for (auto _ : state) {
    RAFT_CUDA_TRY(cudaEventRecord(start_ev, stream));
    cuvs::distance::fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IdxT, DataT>, IdxT>(
      min_out.data(),
      x.data(),
      y.data(),
      xn.data(),
      yn.data(),
      m,
      n,
      k,
      workspace.data(),
      sqrt,
      true,
      true,
      metric,
      0.0f,
      stream);
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

// KVP output benchmarks (standard nearest-neighbor: returns index + distance)
const auto kBenchArgsM = std::vector<int64_t>{1024, 4096, 16384, 65536};
const auto kBenchArgsN = std::vector<int64_t>{32, 64, 128, 256};
const auto kBenchArgsK = std::vector<int64_t>{16, 32, 64, 128, 256};

template <typename DataT, typename IdxT>
void register_fused_nn_benchmark(const std::string& name,
                                 cuvs::distance::DistanceType metric,
                                 bool sqrt)
{
  auto type_name = std::string(typeid(DataT).name()) + "_" + typeid(IdxT).name();
  benchmark::RegisterBenchmark("fusedDistanceNN/" + name + "_" + type_name,
                               [metric, sqrt](benchmark::State& state) {
                                 auto m = static_cast<int64_t>(state.range(0));
                                 auto n = static_cast<int64_t>(state.range(1));
                                 auto k = static_cast<int64_t>(state.range(2));
                                 run_fused_distance_nn_bench<DataT, IdxT>(
                                   state, metric, m, n, k, sqrt);
                               })
    ->ArgsProduct({kBenchArgsM, kBenchArgsN, kBenchArgsK})
    ->Unit(benchmark::kMillisecond);
}

// Skinny-K benchmarks (K < 32, exercises different kernel policy)
const auto kSkinnyBenchArgsM = std::vector<int64_t>{4096, 16384, 65536, 262144};
const auto kSkinnyBenchArgsN = std::vector<int64_t>{32, 64, 128, 256};
const auto kSkinnyBenchArgsK = std::vector<int64_t>{2, 4, 8, 16, 24};

template <typename DataT, typename IdxT>
void register_fused_nn_skinny_benchmark(const std::string& name,
                                        cuvs::distance::DistanceType metric,
                                        bool sqrt)
{
  auto type_name = std::string(typeid(DataT).name()) + "_" + typeid(IdxT).name();
  benchmark::RegisterBenchmark("fusedDistanceNN_skinnyK/" + name + "_" + type_name,
                               [metric, sqrt](benchmark::State& state) {
                                 auto m = static_cast<int64_t>(state.range(0));
                                 auto n = static_cast<int64_t>(state.range(1));
                                 auto k = static_cast<int64_t>(state.range(2));
                                 run_fused_distance_nn_bench<DataT, IdxT>(
                                   state, metric, m, n, k, sqrt);
                               })
    ->ArgsProduct({kSkinnyBenchArgsM, kSkinnyBenchArgsN, kSkinnyBenchArgsK})
    ->Unit(benchmark::kMillisecond);
}

// Large-M benchmarks (realistic k-means scale: millions of samples)
const auto kLargeBenchArgsM = std::vector<int64_t>{1 << 20, 1 << 22};
const auto kLargeBenchArgsN = std::vector<int64_t>{32, 64};
const auto kLargeBenchArgsK = std::vector<int64_t>{32, 64};

template <typename DataT, typename IdxT>
void register_fused_nn_large_benchmark(const std::string& name,
                                       cuvs::distance::DistanceType metric,
                                       bool sqrt)
{
  auto type_name = std::string(typeid(DataT).name()) + "_" + typeid(IdxT).name();
  benchmark::RegisterBenchmark("fusedDistanceNN_large/" + name + "_" + type_name,
                               [metric, sqrt](benchmark::State& state) {
                                 auto m = static_cast<int64_t>(state.range(0));
                                 auto n = static_cast<int64_t>(state.range(1));
                                 auto k = static_cast<int64_t>(state.range(2));
                                 run_fused_distance_nn_bench<DataT, IdxT>(
                                   state, metric, m, n, k, sqrt);
                               })
    ->ArgsProduct({kLargeBenchArgsM, kLargeBenchArgsN, kLargeBenchArgsK})
    ->Unit(benchmark::kMillisecond);
}

void register_all_benchmarks()
{
  // Standard KVP output benchmarks (float, int index)
  register_fused_nn_benchmark<float, int>(
    "L2Expanded", cuvs::distance::DistanceType::L2Expanded, false);
  register_fused_nn_benchmark<float, int>(
    "L2SqrtExpanded", cuvs::distance::DistanceType::L2SqrtExpanded, true);
  register_fused_nn_benchmark<float, int>(
    "CosineExpanded", cuvs::distance::DistanceType::CosineExpanded, false);

  // Int64 index type
  register_fused_nn_benchmark<float, int64_t>(
    "L2Expanded", cuvs::distance::DistanceType::L2Expanded, false);

  // Skinny-K benchmarks (K < 32)
  register_fused_nn_skinny_benchmark<float, int>(
    "L2Expanded", cuvs::distance::DistanceType::L2Expanded, false);
  register_fused_nn_skinny_benchmark<float, int>(
    "CosineExpanded", cuvs::distance::DistanceType::CosineExpanded, false);

  // Large-M benchmarks (production-scale k-means)
  register_fused_nn_large_benchmark<float, int>(
    "L2Expanded", cuvs::distance::DistanceType::L2Expanded, false);
  register_fused_nn_large_benchmark<float, int>(
    "L2SqrtExpanded", cuvs::distance::DistanceType::L2SqrtExpanded, true);
  register_fused_nn_large_benchmark<float, int>(
    "CosineExpanded", cuvs::distance::DistanceType::CosineExpanded, false);
}

}  // namespace

int main(int argc, char** argv)
{
  register_all_benchmarks();

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
