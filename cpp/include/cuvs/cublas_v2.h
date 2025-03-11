// MIT License
// 
// Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#define HIPBLAS_V2
#include <hipblas/hipblas.h>

// Types
#define cublasComputeType_t hipblasComputeType_t
#define cublasDiagType_t hipblasDiagType_t
#define cublasFillMode_t hipblasFillMode_t
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasPointerMode_t hipblasPointerMode_t
#define cublasSideMode_t hipblasSideMode_t
#define cublasStatus_t hipblasStatus_t

// Macros, constants, enums
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32I HIPBLAS_COMPUTE_32I 
#define CUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_POINTER_MODE_DEVICE HIPBLAS_POINTER_MODE_DEVICE
#define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
// TODO(FIXME/HIP): HIPBLAS_STATUS_LICENSE_ERROR is not supported (?)
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_LICENSE_ERROR
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT

// FIXME(HIP/AMD): Using canonical HIP_R_*F would result in passing hipDatatype_t type
// to hipblasDotEx, but hipblasDotEx expects hipblasDatatype_t.
#define CUDA_R_32F HIPBLAS_R_32F
#define CUDA_R_64F HIPBLAS_R_64F

// Functions
#define cublasCreate hipblasCreate
#define cublasDaxpy hipblasDaxpy
#define cublasDcopy hipblasDcopy
#define cublasDestroy hipblasDestroy
#define cublasDgeam hipblasDgeam
#define cublasDgelsBatched hipblasDgelsBatched
#define cublasDgemm hipblasDgemm
#define cublasDgemmBatched hipblasDgemmBatched
#define cublasDgemmStridedBatched hipblasDgemmStridedBatched
#define cublasDgemv hipblasDgemv
#define cublasDger  hipblasDger
#define cublasDgetrfBatched hipblasDgetrfBatched
#define cublasDgetriBatched hipblasDgetriBatched
#define cublasDnrm2 hipblasDnrm2
#define cublasDotEx hipblasDotEx
#define cublasDscal hipblasDscal
#define cublasDswap hipblasDswap
#define cublasDsymm hipblasDsymm
#define cublasDsyrk hipblasDsyrk
#define cublasDtrsm hipblasDtrsm
#define cublasSaxpy hipblasSaxpy
#define cublasScopy hipblasScopy
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasSetStream hipblasSetStream
#define cublasSgeam hipblasSgeam
#define cublasSgelsBatched hipblasSgelsBatched
#define cublasSgemm hipblasSgemm
#define cublasSgemmBatched hipblasSgemmBatched
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasSgemv hipblasSgemv
#define cublasSger  hipblasSger
#define cublasSgetrfBatched hipblasSgetrfBatched
#define cublasSgetriBatched hipblasSgetriBatched
#define cublasSnrm2 hipblasSnrm2
#define cublasSscal hipblasSscal
#define cublasSswap hipblasSswap
#define cublasSsymm hipblasSsymm
#define cublasSsyrk hipblasSsyrk
#define cublasStrsm hipblasStrsm