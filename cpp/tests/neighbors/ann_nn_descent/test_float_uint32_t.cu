/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

// (HIP/AMD) Note: These tests were recently re-enabled upstream as part of
// https://github.com/rapidsai/cuvs/commit/bd6d4a9934ad7f3f3cf2e2c6446abdbdf10ffba0
typedef AnnNNDescentTest<float, float, std::uint32_t> AnnNNDescentTestF_U32;
TEST_P(AnnNNDescentTestF_U32, AnnNNDescent) { this->testNNDescent(); }

typedef AnnNNDescentDistEpiTest<float, float, std::uint32_t> AnnNNDescentTestDistEpiF_U32;
TEST_P(AnnNNDescentTestDistEpiF_U32, AnnNNDescentDistEpi) { this->testNNDescent(); }

INSTANTIATE_TEST_CASE_P(AnnNNDescentTest, AnnNNDescentTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnNNDescentDistEpi,
                        AnnNNDescentTestDistEpiF_U32,
                        ::testing::ValuesIn(inputsDistEpilogue));

}  // namespace   cuvs::neighbors::nn_descent
