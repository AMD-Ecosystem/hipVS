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

// IVF benchmark-realistic sizes: (m, n, k) matching actual fusedDistanceNN calls
// during IVF_FLAT and IVF_PQ index builds via kmeans_balanced.
//
// In the IVF build path the call is always <float, KVP<int64_t, float>, int64_t>.
//
// Datasets and their typical parameters:
//   sift-128-euclidean:       1M rows,   dim=128, nlist=1024-8192,  L2
//   glove-100-angular:        1.1M rows, dim=100, nlist=1024-8192,  Cosine
//   deep-image-96-angular:    10M rows,  dim=96,  nlist=1024-64000, Cosine
//   gist-960-euclidean:       1M rows,   dim=960, nlist=1024-8192,  L2
//   wiki_all-768:             1-10M rows, dim=768, nlist=1024-65536, L2
//   fashion/mnist-784:        60K rows,  dim=784, nlist=1024,       L2
//
// m = training subset size (n_rows / trainset_ratio), typically 250K - 10M
// n = nlist (number of centroids), 1024 - 65536
// k = dataset dimensionality
//
// Each benchmark entry is {m, n, k}. We pick representative combos rather than
// a full product to keep runtime manageable.
// clang-format off
const auto kIvfBenchArgs = std::vector<std::array<int64_t, 3>>{
  // sift-128-euclidean: 1M rows, ratio 1-4 -> m=250K-1M, nlist=1024-8192
  {1000000, 1024, 128},
  {1000000, 4096, 128},
  {1000000, 8192, 128},
  { 250000, 1024, 128},
  { 250000, 4096, 128},
  // glove-100-angular: 1.1M rows, nlist=1024-4096
  {1000000, 1024, 100},
  {1000000, 4096, 100},
  { 275000, 4096, 100},
  // deep-image-96: 10M rows, ratio 1-4 -> m=2.5M-10M, nlist up to 64000
  {2500000, 4096,  96},
  {2500000, 16384, 96},
  {2500000, 64000, 96},
  // gist-960-euclidean: 1M rows, nlist=1024-4096
  {1000000, 1024, 960},
  {1000000, 4096, 960},
  { 250000, 1024, 960},
  // wiki_all-768: 1-10M rows, nlist=1024-65536
  {1000000, 1024, 768},
  {1000000, 8192, 768},
  {2500000, 8192, 768},
  // fashion/mnist-784-euclidean: 60K rows, nlist=1024
  {  60000, 1024, 784},
};
// clang-format on

template <typename DataT, typename IdxT>
void register_fused_nn_ivf_benchmark(const std::string& name,
                                     cuvs::distance::DistanceType metric,
                                     bool sqrt)
{
  auto type_name = std::string(typeid(DataT).name()) + "_" + typeid(IdxT).name();
  for (size_t i = 0; i < kIvfBenchArgs.size(); ++i) {
    auto m          = kIvfBenchArgs[i][0];
    auto n          = kIvfBenchArgs[i][1];
    auto k          = kIvfBenchArgs[i][2];
    auto bench_name = "fusedDistanceNN_ivf/" + name + "_" + type_name + "/" + std::to_string(m) +
                      "/" + std::to_string(n) + "/" + std::to_string(k);
    benchmark::RegisterBenchmark(bench_name, [metric, sqrt, m, n, k](benchmark::State& state) {
      run_fused_distance_nn_bench<DataT, IdxT>(state, metric, m, n, k, sqrt);
    })->Unit(benchmark::kMillisecond);
  }
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

  // IVF-realistic benchmarks: <float, int64_t> matching the actual IVF_FLAT/IVF_PQ
  // build path (kmeans_balanced always uses float data and int64_t indices)
  register_fused_nn_ivf_benchmark<float, int64_t>(
    "L2Expanded", cuvs::distance::DistanceType::L2Expanded, false);
  register_fused_nn_ivf_benchmark<float, int64_t>(
    "L2SqrtExpanded", cuvs::distance::DistanceType::L2SqrtExpanded, true);
  register_fused_nn_ivf_benchmark<float, int64_t>(
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
