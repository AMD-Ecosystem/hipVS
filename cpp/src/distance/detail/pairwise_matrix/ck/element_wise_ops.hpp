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

// Wrapper for norm-based epilogue: distance op (e, c, norm_x, norm_y) + optional sqrt + fin_op.
// Uses existing ck_ops (l2_exp_ck_op, cosine_ck_op) which now have norm-based operator() overloads.

#pragma once

#include <ck_tile/core.hpp>
#include <raft/core/math.hpp>

namespace cuvs::distance::detail {

// Combined epilogue op: distance from (c, norm_x, norm_y) + optional sqrt + fin_op.
// DistanceOpT must have operator()(e, c, norm_x, norm_y) - e.g. l2_exp_ck_op, cosine_ck_op.
template <typename DistanceOpT, typename FinOpT, bool ApplySqrtToDistances = false>
struct OperatorWrapper {
  static_assert(std::is_default_constructible_v<DistanceOpT>,
                "CK epilogue requires a default-constructible distance op");
  static_assert(std::is_default_constructible_v<FinOpT>,
                "CK epilogue requires a default-constructible fin_op");

  FinOpT m_fin_op{};
  DistanceOpT m_distance_op{};

  OperatorWrapper() = default;
  OperatorWrapper(FinOpT fin_op, DistanceOpT distance_op)
    : m_fin_op(fin_op), m_distance_op(distance_op)
  {
  }

  template <typename E, typename C, typename Nx, typename Ny>
  __host__ __device__ void operator()(E& e, const C& c, const Nx& norm_x, const Ny& norm_y) const
  {
    E result{};
    m_distance_op(result, c, norm_x, norm_y);
    if constexpr (ApplySqrtToDistances) { result = raft::sqrt(result); }
    e = m_fin_op(result, 0);
  }
};

}  // namespace cuvs::distance::detail
