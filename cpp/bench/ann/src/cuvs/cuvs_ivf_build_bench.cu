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

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

namespace cuvs::neighbors::ivf_build_bench {

// ---------------------------------------------------------------------------
// Dataset parameters
// ---------------------------------------------------------------------------

constexpr int64_t kNumVecs         = 10'000'000;
constexpr int64_t kDim             = 128;
constexpr int64_t kDatasetClusters = 1024;
constexpr float kClusterStddev     = 0.1f;

constexpr int kWarmupIterations = 1;

// ---------------------------------------------------------------------------
// RAII wrapper for a pair of CUDA events used for GPU timing
// ---------------------------------------------------------------------------

struct cuda_event_pair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop  = nullptr;

  cuda_event_pair()
  {
    RAFT_CUDA_TRY(cudaEventCreate(&start));
    RAFT_CUDA_TRY(cudaEventCreate(&stop));
  }

  ~cuda_event_pair()
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(start));
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(stop));
  }

  cuda_event_pair(const cuda_event_pair&)            = delete;
  cuda_event_pair& operator=(const cuda_event_pair&) = delete;

  // Record start, invoke f(), record stop, synchronize, return elapsed ms.
  template <typename Func>
  float record(cudaStream_t stream, Func&& f)
  {
    RAFT_CUDA_TRY(cudaEventRecord(start, stream));
    f();
    RAFT_CUDA_TRY(cudaEventRecord(stop, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(stop));
    float ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const char* metric_name(cuvs::distance::DistanceType m)
{
  switch (m) {
    case cuvs::distance::DistanceType::L2Expanded: return "L2";
    case cuvs::distance::DistanceType::InnerProduct: return "IP";
    case cuvs::distance::DistanceType::CosineExpanded: return "Cosine";
    default: return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// IVF-Flat build benchmark
// ---------------------------------------------------------------------------

void run_ivf_flat_build(benchmark::State& state,
                        const float* database,
                        uint32_t n_lists,
                        uint32_t kmeans_n_iters,
                        uint32_t ratio,
                        cuvs::distance::DistanceType metric)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvs::neighbors::ivf_flat::index_params index_params;
  index_params.metric                   = metric;
  index_params.n_lists                  = n_lists;
  index_params.kmeans_n_iters           = kmeans_n_iters;
  index_params.kmeans_trainset_fraction = 1.0 / static_cast<double>(ratio);

  auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(database, kNumVecs, kDim);

  for (int i = 0; i < kWarmupIterations; ++i) {
    auto idx = cuvs::neighbors::ivf_flat::build(handle, index_params, dataset_view);
    raft::resource::sync_stream(handle, stream);
  }

  cuda_event_pair events;
  double gpu_time_total_ms = 0.0;
  for (auto _ : state) {
    gpu_time_total_ms += events.record(stream, [&] {
      auto idx = cuvs::neighbors::ivf_flat::build(handle, index_params, dataset_view);
    });
  }

  state.counters["GPU time (ms)"] =
    benchmark::Counter(gpu_time_total_ms, benchmark::Counter::kAvgIterations);
  state.counters["num_vecs"] = benchmark::Counter(static_cast<double>(kNumVecs));
  state.counters["dim"]      = benchmark::Counter(static_cast<double>(kDim));
  state.counters["n_lists"]  = benchmark::Counter(static_cast<double>(n_lists));
  state.counters["niter"]    = benchmark::Counter(static_cast<double>(kmeans_n_iters));
  state.counters["ratio"]    = benchmark::Counter(static_cast<double>(ratio));
}

// ---------------------------------------------------------------------------
// IVF-PQ build benchmark
// ---------------------------------------------------------------------------

void run_ivf_pq_build(benchmark::State& state,
                      const float* database,
                      uint32_t n_lists,
                      uint32_t kmeans_n_iters,
                      uint32_t ratio,
                      uint32_t pq_dim,
                      uint32_t pq_bits,
                      cuvs::distance::DistanceType metric)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  cuvs::neighbors::ivf_pq::index_params index_params;
  index_params.metric                   = metric;
  index_params.n_lists                  = n_lists;
  index_params.kmeans_n_iters           = kmeans_n_iters;
  index_params.kmeans_trainset_fraction = 1.0 / static_cast<double>(ratio);
  index_params.pq_dim                   = pq_dim;
  index_params.pq_bits                  = pq_bits;
  index_params.codebook_kind            = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;

  auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(database, kNumVecs, kDim);

  for (int i = 0; i < kWarmupIterations; ++i) {
    auto idx = cuvs::neighbors::ivf_pq::build(handle, index_params, dataset_view);
    raft::resource::sync_stream(handle, stream);
  }

  cuda_event_pair events;
  double gpu_time_total_ms = 0.0;
  for (auto _ : state) {
    gpu_time_total_ms += events.record(stream, [&] {
      auto idx = cuvs::neighbors::ivf_pq::build(handle, index_params, dataset_view);
    });
  }

  state.counters["GPU time (ms)"] =
    benchmark::Counter(gpu_time_total_ms, benchmark::Counter::kAvgIterations);
  state.counters["num_vecs"] = benchmark::Counter(static_cast<double>(kNumVecs));
  state.counters["dim"]      = benchmark::Counter(static_cast<double>(kDim));
  state.counters["n_lists"]  = benchmark::Counter(static_cast<double>(n_lists));
  state.counters["niter"]    = benchmark::Counter(static_cast<double>(kmeans_n_iters));
  state.counters["ratio"]    = benchmark::Counter(static_cast<double>(ratio));
  state.counters["pq_dim"]   = benchmark::Counter(static_cast<double>(pq_dim));
  state.counters["pq_bits"]  = benchmark::Counter(static_cast<double>(pq_bits));
}

// ---------------------------------------------------------------------------
// Benchmark registration
//
// Parameter grids are modelled after the "base" groups in:
//   python/cuvs_bench/cuvs_bench/config/algos/cuvs_ivf_flat.yaml
//   python/cuvs_bench/cuvs_bench/config/algos/cuvs_ivf_pq.yaml
// ---------------------------------------------------------------------------

// cuvs_ivf_flat.yaml "base":
//   nlist:  [1024, 2048, 4096, 8192, 16384, 32000, 64000]
//   ratio:  [1, 2, 4]
//   niter:  [20, 25]
void register_ivf_flat_benchmarks(const float* database)
{
  constexpr auto metric = cuvs::distance::DistanceType::L2Expanded;

  for (uint32_t nlist : {1024u, 2048u, 4096u, 8192u, 16384u, 32000u, 64000u}) {
    for (uint32_t ratio : {1u, 2u, 4u}) {
      for (uint32_t niter : {20u, 25u}) {
        auto name = std::string("ivf_flat_build/") + metric_name(metric) + "/nlist_" +
                    std::to_string(nlist) + "/ratio_" + std::to_string(ratio) + "/niter_" +
                    std::to_string(niter);

        benchmark::RegisterBenchmark(name,
                                     [database, nlist, niter, ratio](benchmark::State& state) {
                                       run_ivf_flat_build(
                                         state, database, nlist, niter, ratio, metric);
                                     })
          ->Unit(benchmark::kMillisecond)
          ->Iterations(1);
      }
    }
  }
}

// cuvs_ivf_pq.yaml "base":
//   nlist:   [1024, 2048, 4096, 8192]
//   pq_dim:  [64, 32, 16]
//   pq_bits: [8, 6, 5, 4]
//   ratio:   [10]
//   niter:   [25]
void register_ivf_pq_benchmarks(const float* database)
{
  constexpr auto metric    = cuvs::distance::DistanceType::L2Expanded;
  constexpr uint32_t ratio = 10;
  constexpr uint32_t niter = 25;

  for (uint32_t nlist : {1024u, 2048u, 4096u, 8192u}) {
    for (uint32_t pq_dim : {64u, 32u, 16u}) {
      for (uint32_t pq_bits : {8u, 6u, 5u, 4u}) {
        auto name = std::string("ivf_pq_build/") + metric_name(metric) + "/nlist_" +
                    std::to_string(nlist) + "/pq_dim_" + std::to_string(pq_dim) + "/pq_bits_" +
                    std::to_string(pq_bits);

        benchmark::RegisterBenchmark(
          name,
          [database, nlist, pq_dim, pq_bits](benchmark::State& state) {
            run_ivf_pq_build(state, database, nlist, niter, ratio, pq_dim, pq_bits, metric);
          })
          ->Unit(benchmark::kMillisecond)
          ->Iterations(1);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

// Generates a clustered dataset using raft::random::make_blobs (GPU equivalent
// of sklearn.datasets.make_blobs).  Called once; the resulting buffer is shared
// read-only across all benchmark cases.
rmm::device_uvector<float> generate_dataset()
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<float> database(kNumVecs * kDim, stream);
  rmm::device_uvector<int64_t> labels(kNumVecs, stream);

  raft::random::make_blobs<float, int64_t>(database.data(),
                                           labels.data(),
                                           kNumVecs,
                                           kDim,
                                           kDatasetClusters,
                                           stream,
                                           /*row_major=*/true,
                                           /*centers=*/nullptr,
                                           /*cluster_std=*/nullptr,
                                           /*cluster_std_scalar=*/kClusterStddev,
                                           /*shuffle=*/true,
                                           /*center_box_min=*/-10.0f,
                                           /*center_box_max=*/10.0f,
                                           /*seed=*/1234ULL);
  raft::resource::sync_stream(handle, stream);
  return database;
}

}  // namespace cuvs::neighbors::ivf_build_bench

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
  using namespace cuvs::neighbors::ivf_build_bench;

  // Dataset is generated once and shared read-only across all benchmark cases.
  auto database = generate_dataset();

  register_ivf_flat_benchmarks(database.data());
  register_ivf_pq_benchmarks(database.data());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
