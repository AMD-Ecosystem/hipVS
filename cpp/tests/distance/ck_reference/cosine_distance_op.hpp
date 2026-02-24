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

// Cosine distance epilogue for CK Tile multi-D GEMM (see pairwise_distance_ck.md and
// rocm-libraries example 19_gemm_multi_d). E = 1 - (a·b) / (||a|| ||b||).

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {
namespace element_wise {

struct CosineDistance {
  static constexpr const char* name = "CosineDistance";

  template <typename E, typename C, typename Nx, typename Ny>
  __host__ __device__ void operator()(E& e, const C& c, const Nx& norm_x, const Ny& norm_y) const
  {
    constexpr float kEpsilon = 1e-8f;

    float dot_product = ck_tile::type_convert<float>(c);
    float norm_a_times_norm_b =
      ck_tile::type_convert<float>(norm_x) * ck_tile::type_convert<float>(norm_y);
    float similarity =
      (norm_a_times_norm_b > kEpsilon) ? (dot_product / norm_a_times_norm_b) : 0.0f;

    e = ck_tile::type_convert<E>(1.0f - similarity);
  }
};

}  // namespace element_wise
}  // namespace ck_tile
