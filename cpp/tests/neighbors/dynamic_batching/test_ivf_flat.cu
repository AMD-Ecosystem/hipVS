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
#include <gtest/gtest.h>

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::dynamic_batching {

using Base = dynamic_batching_test<uint8_t,
                                   int64_t,
                                   ivf_flat::index<uint8_t, int64_t>,
                                   ivf_flat::build,
                                   ivf_flat::search>;
struct ivf_flat_i8 : Base {
  void SetUp() override { Base::SetUp(); }
};

TEST_P(ivf_flat_i8, defaults)
{
  build_params_upsm.n_lists = std::round(std::sqrt(ps.n_rows));
  search_params_upsm.n_probes =
    std::max<uint32_t>(std::min<uint32_t>(build_params_upsm.n_lists, 10),
                       raft::div_rounding_up_safe<uint32_t>(build_params_upsm.n_lists, 50));
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, ivf_flat_i8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
