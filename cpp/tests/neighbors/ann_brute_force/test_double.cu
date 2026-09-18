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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../ann_brute_force.cuh"

namespace cuvs::neighbors::brute_force {

using AnnBruteForceTest_double = AnnBruteForceTest<double, double, int64_t>;
TEST_P(AnnBruteForceTest_double, AnnBruteForce) { this->testBruteForce(); }

// Exercise alignment, large dimensions, and both sides of the wavefront and fused-k limits.
const std::vector<AnnBruteForceInputs<int64_t>> double_inputs = {
  {17, 257, 1, 1, cuvs::distance::DistanceType::L2Expanded},
  {17, 257, 3, 31, cuvs::distance::DistanceType::L2Unexpanded},
  {17, 257, 4, 32, cuvs::distance::DistanceType::L2SqrtExpanded},
  {17, 257, 8, 33, cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {17, 257, 16, 64, cuvs::distance::DistanceType::L2Expanded},
  {17, 257, 32, 65, cuvs::distance::DistanceType::L2Unexpanded},
  {17, 257, 2049, 16, cuvs::distance::DistanceType::L2Expanded},
  {17, 257, 2050, 65, cuvs::distance::DistanceType::InnerProduct}};

INSTANTIATE_TEST_SUITE_P(AnnBruteForceTest,
                         AnnBruteForceTest_double,
                         ::testing::ValuesIn(double_inputs));
}  // namespace cuvs::neighbors::brute_force
