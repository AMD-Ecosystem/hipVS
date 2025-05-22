/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "detail/dynamic_batching.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::dynamic_batching {

// NB: the (template) index parameter should be the last; it may contain the spaces and so split
//       into multiple preprocessor token. Then it is consumed as __VA_ARGS__
//
#define CUVS_INST_DYNAMIC_BATCHING_INDEX(T, IdxT, Namespace, ...)                         \
  template <>                                                                             \
  template <>                                                                             \
  index<T, IdxT>::index(                                                                  \
    const raft::resources& res,                                                           \
    const cuvs::neighbors::dynamic_batching::index_params& params,                        \
    const Namespace ::__VA_ARGS__& upstream_index,                                        \
    const typename Namespace ::__VA_ARGS__::search_params_type& upstream_params,          \
    const cuvs::neighbors::filtering::base_filter* sample_filter)                         \
    : runner{new detail::batch_runner<T, IdxT>(                                           \
        res, params, upstream_index, upstream_params, Namespace ::search, sample_filter)} \
  {                                                                                       \
  }

#define CUVS_INST_DYNAMIC_BATCHING_SEARCH(T, IdxT)                                 \
  void search(raft::resources const& res,                                          \
              cuvs::neighbors::dynamic_batching::search_params const& params,      \
              cuvs::neighbors::dynamic_batching::index<T, IdxT> const& index,      \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries, \
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,  \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances) \
  {                                                                                \
    return index.runner->search(res, params, queries, neighbors, distances);       \
  }

CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::brute_force, index<float, float>);

CUVS_INST_DYNAMIC_BATCHING_INDEX(float, uint32_t, cuvs::neighbors::cagra, index<float, uint32_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, uint32_t, cuvs::neighbors::cagra, index<half, uint32_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, uint32_t, cuvs::neighbors::cagra, index<int8_t, uint32_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t,
                                 uint32_t,
                                 cuvs::neighbors::cagra,
                                 index<uint8_t, uint32_t>);
#ifdef __HIP_PLATFORM_AMD__
#define CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(T, IdxT, Namespace, ...)             \
  template <>                                                                             \
  template <>                                                                             \
  index<T, IdxT>::index(                                                                  \
    const raft::resources& res,                                                           \
    const cuvs::neighbors::dynamic_batching::index_params& params,                        \
    const Namespace ::__VA_ARGS__& upstream_index,                                        \
    const typename Namespace ::__VA_ARGS__::search_params_type& upstream_params,          \
    const cuvs::neighbors::filtering::base_filter* sample_filter)                         \
    : runner{new detail::batch_runner<T, IdxT>(                                           \
        res, params, upstream_index, upstream_params, Namespace ::search, sample_filter)} \
  {                                                                                       \
    RAFT_FAIL("Unsupported upstream index used");                                         \
  }

CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(float,
                                             int64_t,
                                             cuvs::neighbors::ivf_pq,
                                             index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(half,
                                             int64_t,
                                             cuvs::neighbors::ivf_pq,
                                             index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(int8_t,
                                             int64_t,
                                             cuvs::neighbors::ivf_pq,
                                             index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(uint8_t,
                                             int64_t,
                                             cuvs::neighbors::ivf_pq,
                                             index<int64_t>);

CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(float,
                                             int64_t,
                                             cuvs::neighbors::ivf_flat,
                                             index<float, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(int8_t,
                                             int64_t,
                                             cuvs::neighbors::ivf_flat,
                                             index<int8_t, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX_UNSUPPORTED(uint8_t,
                                             int64_t,
                                             cuvs::neighbors::ivf_flat,
                                             index<uint8_t, int64_t>);

#else
CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);

CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::ivf_flat, index<float, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t,
                                 int64_t,
                                 cuvs::neighbors::ivf_flat,
                                 index<int8_t, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t,
                                 int64_t,
                                 cuvs::neighbors::ivf_flat,
                                 index<uint8_t, int64_t>);
#endif
CUVS_INST_DYNAMIC_BATCHING_SEARCH(float, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(half, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(int8_t, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(uint8_t, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(float, uint32_t);  // uint32_t index type is needed for CAGRA
CUVS_INST_DYNAMIC_BATCHING_SEARCH(half, uint32_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(int8_t, uint32_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(uint8_t, uint32_t);

#undef CUVS_INST_DYNAMIC_BATCHING_INDEX
#undef CUVS_INST_DYNAMIC_BATCHING_SEARCH

}  // namespace cuvs::neighbors::dynamic_batching
