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

// Compile-time selection between the two ck_tile GEMM APIs the distance CK
// kernels are written against.
//
// The header-only ck_tile GEMM API changed between ROCm 7.2.x and ROCm 7.13:
//   * ck_tile::UniversalGemmPipelineProblem dropped two template parameters
//     (a leading bool and a ck_tile::TailNumber).
//   * ck_tile::UniversalGemmKernel replaced the
//     MakeGemmTensorViews -> MakeGemmPadViews -> MakeGemmTileWindows view
//     pipeline with per-operand MakeABlockWindows / MakeBBlockWindows /
//     MakeCBlockWindows helpers.
//
// CUVS_CK_TILE_NEW_GEMM_API == 1 selects the ROCm 7.13+ API; 0 selects the
// ROCm 7.2.x API. The boundary is set at 7.13 (the version known to ship the
// new API); intermediate releases (7.3 - 7.12) are not separately targeted and
// fall back to the 7.2.x path.

#pragma once

#include <rocm-core/rocm_version.h>

#if ROCM_VERSION_MAJOR > 7 || (ROCM_VERSION_MAJOR == 7 && ROCM_VERSION_MINOR >= 13)
#define CUVS_CK_TILE_NEW_GEMM_API 1
#else
#define CUVS_CK_TILE_NEW_GEMM_API 0
#endif
