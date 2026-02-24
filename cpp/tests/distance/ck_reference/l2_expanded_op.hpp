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

// L2Expanded epilogue for CK Tile multi-D GEMM.
// E = ||x||^2 + ||y||^2 - 2 * (x·y).

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {
namespace element_wise {

struct L2Expanded {
  static constexpr const char* name = "L2Expanded";

  template <typename E, typename C, typename Nx, typename Ny>
  __host__ __device__ void operator()(E& e, const C& c, const Nx& norm_x, const Ny& norm_y) const
  {
    float dot_product = ck_tile::type_convert<float>(c);
    float norm_sq_sum = ck_tile::type_convert<float>(norm_x) + ck_tile::type_convert<float>(norm_y);

    e = ck_tile::type_convert<E>(norm_sq_sum - 2.0f * dot_product);
  }
};

}  // namespace element_wise
}  // namespace ck_tile
