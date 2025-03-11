// MIT License
//
// Copyright (c) 2024 -2025 Advanced Micro Devices, Inc.
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

#include <hip/library_types.h>

#define cudaDataType hipDataType

#define CUDA_R_16F  HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16I  HIP_R_16I
#define CUDA_R_16U  HIP_R_16U
#define CUDA_R_32F  HIP_R_32F
#define CUDA_R_32I  HIP_R_32I
#define CUDA_R_32U  HIP_R_32U
#define CUDA_R_64I  HIP_R_64I
#define CUDA_R_64U  HIP_R_64U
#define CUDA_R_8I   HIP_R_8I
#define CUDA_R_8U   HIP_R_8U
#define CUDA_R_4I   HIP_R_4I
#define CUDA_R_4U   HIP_R_4U

#define CUDA_C_16F  HIP_C_16F
#define CUDA_C_16BF HIP_C_16BF
#define CUDA_C_16I  HIP_C_16I
#define CUDA_C_16U  HIP_C_16U
#define CUDA_C_32F  HIP_C_32F
#define CUDA_C_32I  HIP_C_32I
#define CUDA_C_32U  HIP_C_32U
#define CUDA_C_64F  HIP_C_64F
#define CUDA_C_64I  HIP_C_64I
#define CUDA_C_64U  HIP_C_64U
#define CUDA_C_8I   HIP_C_8I
#define CUDA_C_8U   HIP_C_8U
#define CUDA_C_4I   HIP_C_4I
#define CUDA_C_4U   HIP_C_4U