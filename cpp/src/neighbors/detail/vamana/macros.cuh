/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
 */
/*
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
#pragma once

namespace cuvs::neighbors::vamana::detail {

/* Macros to compute the shared memory requirements for CUB primitives used by search and prune */
#define COMPUTE_SMEM_SIZES(degree, visited_size, DEG, CANDS)                                      \
  if (degree == DEG && visited_size <= CANDS && visited_size > CANDS / 2) {                       \
    if (warp_size == 32) {                                                                        \
      sort_smem_size = static_cast<int>(                                                          \
        sizeof(typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, CANDS / 32>::TempStorage)); \
    } else if (warp_size == 64) {                                                                 \
      sort_smem_size = static_cast<int>(                                                          \
        sizeof(typename cub::BlockMergeSort<DistPair<IdxT, accT>, 64, CANDS / 64>::TempStorage)); \
    } else {                                                                                      \
      RAFT_FAIL("Invalid warp size: %d", warp_size);                                              \
    }                                                                                             \
  }

// Current supported sizes for degree and visited_size. Note that visited_size must be > degree
#define SELECT_SORT_SMEM_SIZE(degree, visited_size)  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 64);   \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 128);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 256);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 32, 512);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 128);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 256);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 64, 512);  \
  COMPUTE_SMEM_SIZE(degree, visited_size, 128, 256); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 128, 512); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 256, 512); \
  COMPUTE_SMEM_SIZE(degree, visited_size, 256, 1024);

/* Macros to call the CUB BlockSort primitives for supported sizes for GREEDY SEARCH */
#define SEARCH_CALL_SORT(topk, CANDS)                                                          \
  if (topk <= CANDS && topk > CANDS / 2) {                                                     \
    using BlockSortT =                                                                         \
      cub::BlockMergeSort<DistPair<IdxT, accT>, raft::warp_size(), CANDS / raft::warp_size()>; \
    auto& sort_mem = reinterpret_cast<typename BlockSortT::TempStorage&>(smem);                \
    sort_visited<accT, IdxT, CANDS>(&query_list[i], &sort_mem);                                \
  }

// SEARCH only relies on visited_size (not degree) for shared memory.
#define SEARCH_SELECT_SORT(topk) \
  SEARCH_CALL_SORT(topk, 64);    \
  SEARCH_CALL_SORT(topk, 128);   \
  SEARCH_CALL_SORT(topk, 256);   \
  SEARCH_CALL_SORT(topk, 512);   \
  SEARCH_CALL_SORT(topk, 1024);

}  // namespace cuvs::neighbors::vamana::detail
