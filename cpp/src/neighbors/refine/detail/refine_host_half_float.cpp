/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "../refine_host.hpp"
#include <cuvs/neighbors/refine.hpp>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

#define instantiate_cuvs_neighbors_refine(idx_t, data_t, distance_t, matrix_idx)          \
  void cuvs::neighbors::refine(                                                           \
    raft::resources const& handle,                                                        \
    raft::host_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,            \
    raft::host_matrix_view<const data_t, matrix_idx, raft::row_major> queries,            \
    raft::host_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates, \
    raft::host_matrix_view<idx_t, matrix_idx, raft::row_major> indices,                   \
    raft::host_matrix_view<distance_t, matrix_idx, raft::row_major> distances,            \
    cuvs::distance::DistanceType metric)                                                  \
  {                                                                                       \
    refine_impl<idx_t, data_t, distance_t, matrix_idx>(                                   \
      handle, dataset, queries, neighbor_candidates, indices, distances, metric);         \
  }

instantiate_cuvs_neighbors_refine(int64_t, half, float, int64_t);

#undef instantiate_cuvs_neighbors_refine
