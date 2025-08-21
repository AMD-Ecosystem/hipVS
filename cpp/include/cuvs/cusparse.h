// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// FIXME(HIP/AMD): hipsparse does not want CUDART_VERSION to be set, as otherwise
// hipSparseStatus_t is not defined
#ifdef CUDART_VERSION
#undef CUDART_VERSION
#endif

#include <hipsparse/hipsparse.h>
#include <raft/cusparse.h>

// types
#ifndef cusparseAction_t
#define cusparseAction_t hipsparseAction_t
#endif
#ifndef cusparseCsr2CscAlg_t
#define cusparseCsr2CscAlg_t hipsparseCsr2CscAlg_t
#endif
#ifndef cusparseDnVecDescr_t
#define cusparseDnVecDescr_t hipsparseDnVecDescr_t
#endif
#ifndef cusparseHandle_t
#define cusparseHandle_t hipsparseHandle_t
#endif
#ifndef cusparseIndexBase_t
#define cusparseIndexBase_t hipsparseIndexBase_t
#endif
#ifndef cusparseMatDescr_t
#define cusparseMatDescr_t hipsparseMatDescr_t
#endif
#ifndef cusparseOperation_t
#define cusparseOperation_t hipsparseOperation_t
#endif
#ifndef cusparsePointerMode_t
#define cusparsePointerMode_t hipsparsePointerMode_t
#endif
#ifndef cusparseSpVecDescr_t
#define cusparseSpVecDescr_t hipsparseSpVecDescr_t
#endif
#ifndef cusparseStatus_t
#define cusparseStatus_t hipsparseStatus_t
#endif
#ifndef cusparseDnMatDescr_t
#define cusparseDnMatDescr_t hipsparseDnMatDescr_t
#endif
#ifndef cusparseSpMatDescr_t
#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#endif
#ifndef cusparseSpMVAlg_t
#define cusparseSpMVAlg_t hipsparseSpMVAlg_t
#endif
#ifndef cusparseSpMMAlg_t
#define cusparseSpMMAlg_t hipsparseSpMMAlg_t
#endif
#ifndef cusparseSDDMMAlg_t
#define cusparseSDDMMAlg_t hipsparseSDDMMAlg_t
#endif
#ifndef cusparseOrder_t
#define cusparseOrder_t hipsparseOrder_t
#endif

// macros, constants, enums
#ifndef CUSPARSE_INDEX_32I
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#endif
#ifndef CUSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#endif
#ifndef CUSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_ALLOC_FAILED HIPSPARSE_STATUS_ALLOC_FAILED
#endif
#ifndef CUSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_ARCH_MISMATCH HIPSPARSE_STATUS_ARCH_MISMATCH
#endif
#ifndef CUSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_EXECUTION_FAILED HIPSPARSE_STATUS_EXECUTION_FAILED
#endif
#ifndef CUSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_INTERNAL_ERROR HIPSPARSE_STATUS_INTERNAL_ERROR
#endif
#ifndef CUSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_INVALID_VALUE HIPSPARSE_STATUS_INVALID_VALUE
#endif
#ifndef CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#endif
#ifndef CUSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_NOT_INITIALIZED HIPSPARSE_STATUS_NOT_INITIALIZED
#endif
#ifndef CUSPARSE_STATUS_SUCCESS
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#endif
#ifndef CUSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#endif
#ifndef CUSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#endif
#ifndef CUSPARSE_POINTER_MODE_HOST
#define CUSPARSE_POINTER_MODE_HOST HIPSPARSE_POINTER_MODE_HOST
#endif
#ifndef CUSPARSE_MATRIX_TYPE_SYMMETRIC
#define CUSPARSE_MATRIX_TYPE_SYMMETRIC HIPSPARSE_MATRIX_TYPE_SYMMETRIC
#endif
#ifndef CUSPARSE_MATRIX_TYPE_GENERAL
#define CUSPARSE_MATRIX_TYPE_GENERAL HIPSPARSE_MATRIX_TYPE_GENERAL
#endif
#ifndef CUSPARSE_ACTION_NUMERIC
#define CUSPARSE_ACTION_NUMERIC HIPSPARSE_ACTION_NUMERIC
#endif
#ifndef CUSPARSE_CSR2CSC_ALG1
#define CUSPARSE_CSR2CSC_ALG1 HIPSPARSE_CSR2CSC_ALG1
#endif
#ifndef CUSPARSE_ORDER_ROW
#define CUSPARSE_ORDER_ROW HIPSPARSE_ORDER_ROW
#endif
#ifndef CUSPARSE_ORDER_COL
#define CUSPARSE_ORDER_COL HIPSPARSE_ORDER_COL
#endif
#ifndef CUSPARSE_SPMM_CSR_ALG2
#define CUSPARSE_SPMM_CSR_ALG2 HIPSPARSE_SPMM_CSR_ALG2
#endif
#ifndef CUSPARSE_SPMM_CSR_ALG1
#define CUSPARSE_SPMM_CSR_ALG1 HIPSPARSE_SPMM_CSR_ALG1
#endif
#ifndef CUSPARSE_SDDMM_ALG_DEFAULT
#define CUSPARSE_SDDMM_ALG_DEFAULT HIPSPARSE_SDDMM_ALG_DEFAULT
#endif
#ifndef CUSPARSE_INDEX_64I
#define CUSPARSE_INDEX_64I HIPSPARSE_INDEX_64I
#endif
#ifndef CUSPARSE_SPARSETODENSE_ALG_DEFAULT
#define CUSPARSE_SPARSETODENSE_ALG_DEFAULT HIPSPARSE_SPARSETODENSE_ALG_DEFAULT
#endif
#ifndef CUSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMV_CSR_ALG1 HIPSPARSE_SPMV_CSR_ALG1
#endif
#ifndef CUSPARSE_SPMV_CSR_ALG2
#define CUSPARSE_SPMV_CSR_ALG2 HIPSPARSE_SPMV_CSR_ALG2
#endif
#ifndef CUSPARSE_SPMV_ALG_DEFAULT
#define CUSPARSE_SPMV_ALG_DEFAULT HIPSPARSE_SPMV_ALG_DEFAULT
#endif

// functions
#ifndef cusparsecoo2csr
#define cusparsecoo2csr hipsparsecoo2csr
#endif
#ifndef cusparseCreate
#define cusparseCreate hipsparseCreate
#endif
#ifndef cusparseCreateMatDescr
#define cusparseCreateMatDescr hipsparseCreateMatDescr
#endif
#ifndef cusparseCreateIdentityPermutation
#define cusparseCreateIdentityPermutation hipsparseCreateIdentityPermutation
#endif
#ifndef cusparseCsr2cscEx2
#define cusparseCsr2cscEx2 hipsparseCsr2cscEx2
#endif
#ifndef cusparseCsr2cscEx2_bufferSize
#define cusparseCsr2cscEx2_bufferSize hipsparseCsr2cscEx2_bufferSize
#endif
#ifndef cusparseDcsr2dense
#define cusparseDcsr2dense hipsparseDcsr2dense
#endif
#ifndef cusparseDcsrmm
#define cusparseDcsrmm hipsparseDcsrmm
#endif
#ifndef cusparseDcsrmv
#define cusparseDcsrmv hipsparseDcsrmv
#endif
#ifndef cusparseDestroy
#define cusparseDestroy hipsparseDestroy
#endif
#ifndef cusparseDestroyDnVec
#define cusparseDestroyDnVec hipsparseDestroyDnVec
#endif
#ifndef cusparseDestroySpVec
#define cusparseDestroySpVec hipsparseDestroySpVec
#endif
#ifndef cusparseDestroyMatDescr
#define cusparseDestroyMatDescr hipsparseDestroyMatDescr
#endif
#ifndef cusparseDgemmi
#define cusparseDgemmi hipsparseDgemmi
#endif
#ifndef cusparseGather
#define cusparseGather hipsparseGather
#endif
#ifndef cusparseScsr2dense
#define cusparseScsr2dense hipsparseScsr2dense
#endif
#ifndef cusparseScsrmm
#define cusparseScsrmm hipsparseScsrmm
#endif
#ifndef cusparseScsrmv
#define cusparseScsrmv hipsparseScsrmv
#endif
#ifndef cusparseSetPointerMode
#define cusparseSetPointerMode hipsparseSetPointerMode
#endif
#ifndef cusparseSetStream
#define cusparseSetStream hipsparseSetStream
#endif
#ifndef cusparseSgemmi
#define cusparseSgemmi hipsparseSgemmi
#endif
#ifndef cusparseXcoo2csr
#define cusparseXcoo2csr hipsparseXcoo2csr
#endif
#ifndef cusparseXcoosort_bufferSizeExt
#define cusparseXcoosort_bufferSizeExt hipsparseXcoosort_bufferSizeExt
#endif
#ifndef cusparseXcoosortByRow
#define cusparseXcoosortByRow hipsparseXcoosortByRow
#endif
#ifndef cusparseXcsr2coo
#define cusparseXcsr2coo hipsparseXcsr2coo
#endif
#ifndef cusparseSetMatType
#define cusparseSetMatType hipsparseSetMatType
#endif
#ifndef cusparseSetMatIndexBase
#define cusparseSetMatIndexBase hipsparseSetMatIndexBase
#endif
#ifndef cusparseDestroySpMat
#define cusparseDestroySpMat hipsparseDestroySpMat
#endif
#ifndef cusparseDestroyDnMat
#define cusparseDestroyDnMat hipsparseDestroyDnMat
#endif
#ifndef cusparseCreateCsr
#define cusparseCreateCsr hipsparseCreateCsr
#endif
#ifndef cusparseCreateDnVec
#define cusparseCreateDnVec hipsparseCreateDnVec
#endif
#ifndef cusparseSpMV_bufferSize
#define cusparseSpMV_bufferSize hipsparseSpMV_bufferSize
#endif
#ifndef cusparseCreateDnMat
#define cusparseCreateDnMat hipsparseCreateDnMat
#endif
#ifndef cusparseSpMV
#define cusparseSpMV hipsparseSpMV
#endif
#ifndef cusparseCreateCsc
#define cusparseCreateCsc hipsparseCreateCsc
#endif
#ifndef cusparseSpMM
#define cusparseSpMM hipsparseSpMM
#endif
#ifndef cusparseSDDMM_bufferSize
#define cusparseSDDMM_bufferSize hipsparseSDDMM_bufferSize
#endif
#ifndef cusparseSDDMM
#define cusparseSDDMM hipsparseSDDMM
#endif
#ifndef cusparseSpMM_bufferSize
#define cusparseSpMM_bufferSize hipsparseSpMM_bufferSize
#endif
#ifndef cusparseSparseToDense
#define cusparseSparseToDense hipsparseSparseToDense
#endif
#ifndef cusparseSparseToDense_bufferSize
#define cusparseSparseToDense_bufferSize hipsparseSparseToDense_bufferSize
#endif
