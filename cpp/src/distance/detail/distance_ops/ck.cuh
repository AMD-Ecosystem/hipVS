
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

#pragma once

#include <type_traits>  // std::false_type
#include <utility>      // std::declval

namespace cuvs::distance::detail::ops {

// Similar patter as the CUTLASS case. We define a named requirement "has_ck_op" that can be used to
// determine if a distance operation has a CK op that can be used to pass to CK. Examples of
// distance operations that satisfy this requirement are cosine_distance_op and l2_exp_distance_op.

// Primary template handles types that do not support CK.
// This pattern is described in:
// https://en.cppreference.com/w/cpp/types/void_t
template <typename, typename = void>
struct has_ck_op : std::false_type {};

// Specialization recognizes types that supports CK
template <typename T>
struct has_ck_op<T, std::void_t<decltype(std::declval<T>().get_ck_op())>> : std::true_type {};

}  // namespace cuvs::distance::detail::ops
