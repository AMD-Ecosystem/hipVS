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

// types
#define cusparseAction_t      hipsparseAction_t
#define cusparseCsr2CscAlg_t  hipsparseCsr2CscAlg_t
#define cusparseDnVecDescr_t  hipsparseDnVecDescr_t
#define cusparseHandle_t      hipsparseHandle_t
#define cusparseIndexBase_t   hipsparseIndexBase_t
#define cusparseMatDescr_t    hipsparseMatDescr_t
#define cusparseOperation_t   hipsparseOperation_t
#define cusparsePointerMode_t hipsparsePointerMode_t
#define cusparseSpVecDescr_t  hipsparseSpVecDescr_t
#define cusparseStatus_t      hipsparseStatus_t
#define cusparseDnMatDescr_t  hipsparseDnMatDescr_t
#define cusparseSpMatDescr_t  hipsparseSpMatDescr_t
#define cusparseSpMVAlg_t     hipsparseSpMVAlg_t
#define cusparseSpMMAlg_t     hipsparseSpMMAlg_t
#define cusparseSDDMMAlg_t    hipsparseSDDMMAlg_t
#define cusparseOrder_t       hipsparseOrder_t

// macros, constants, enums
#define CUSPARSE_INDEX_32I                        HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_BASE_ZERO                  HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_STATUS_ALLOC_FAILED              HIPSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_ARCH_MISMATCH             HIPSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_EXECUTION_FAILED          HIPSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_INTERNAL_ERROR            HIPSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_INVALID_VALUE             HIPSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_NOT_INITIALIZED           HIPSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_SUCCESS                   HIPSPARSE_STATUS_SUCCESS
#define CUSPARSE_OPERATION_TRANSPOSE              HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_OPERATION_NON_TRANSPOSE          HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_POINTER_MODE_HOST                HIPSPARSE_POINTER_MODE_HOST
#define CUSPARSE_MATRIX_TYPE_SYMMETRIC            HIPSPARSE_MATRIX_TYPE_SYMMETRIC
#define CUSPARSE_MATRIX_TYPE_GENERAL              HIPSPARSE_MATRIX_TYPE_GENERAL
#define CUSPARSE_ACTION_NUMERIC                   HIPSPARSE_ACTION_NUMERIC
#define CUSPARSE_CSR2CSC_ALG1                     HIPSPARSE_CSR2CSC_ALG1
#define CUSPARSE_ORDER_ROW                        HIPSPARSE_ORDER_ROW
#define CUSPARSE_ORDER_COL                        HIPSPARSE_ORDER_COL
#define CUSPARSE_SPMM_CSR_ALG2                    HIPSPARSE_SPMM_CSR_ALG2
#define CUSPARSE_SPMM_CSR_ALG1                    HIPSPARSE_SPMM_CSR_ALG1
#define CUSPARSE_SDDMM_ALG_DEFAULT                HIPSPARSE_SDDMM_ALG_DEFAULT
#define CUSPARSE_INDEX_64I                        HIPSPARSE_INDEX_64I
#define CUSPARSE_SPARSETODENSE_ALG_DEFAULT        HIPSPARSE_SPARSETODENSE_ALG_DEFAULT
#define CUSPARSE_SPMV_CSR_ALG1                    HIPSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMV_CSR_ALG2                    HIPSPARSE_SPMV_CSR_ALG2
#define CUSPARSE_SPMV_ALG_DEFAULT                 HIPSPARSE_SPMV_ALG_DEFAULT

// functions
#define cusparsecoo2csr                   hipsparsecoo2csr
#define cusparseCreate                    hipsparseCreate
#define cusparseCreateMatDescr            hipsparseCreateMatDescr
#define cusparseCreateIdentityPermutation hipsparseCreateIdentityPermutation
#define cusparseCsr2cscEx2                hipsparseCsr2cscEx2
#define cusparseCsr2cscEx2_bufferSize     hipsparseCsr2cscEx2_bufferSize
#define cusparseDcsr2dense                hipsparseDcsr2dense
#define cusparseDcsrmm                    hipsparseDcsrmm
#define cusparseDcsrmv                    hipsparseDcsrmv
#define cusparseDestroy                   hipsparseDestroy
#define cusparseDestroyDnVec              hipsparseDestroyDnVec
#define cusparseDestroySpVec              hipsparseDestroySpVec
#define cusparseDestroyMatDescr           hipsparseDestroyMatDescr
#define cusparseDgemmi                    hipsparseDgemmi
#define cusparseGather                    hipsparseGather
#define cusparseScsr2dense                hipsparseScsr2dense
#define cusparseScsrmm                    hipsparseScsrmm
#define cusparseScsrmv                    hipsparseScsrmv
#define cusparseSetPointerMode            hipsparseSetPointerMode
#define cusparseSetStream                 hipsparseSetStream
#define cusparseSgemmi                    hipsparseSgemmi
#define cusparseXcoo2csr                  hipsparseXcoo2csr
#define cusparseXcoosort_bufferSizeExt    hipsparseXcoosort_bufferSizeExt
#define cusparseXcoosortByRow             hipsparseXcoosortByRow
#define cusparseXcsr2coo                  hipsparseXcsr2coo
#define cusparseSetMatType                hipsparseSetMatType
#define cusparseSetMatIndexBase           hipsparseSetMatIndexBase
#define cusparseDestroySpMat              hipsparseDestroySpMat
#define cusparseDestroyDnMat              hipsparseDestroyDnMat
#define cusparseCreateCsr                 hipsparseCreateCsr
#define cusparseCreateDnVec               hipsparseCreateDnVec
#define cusparseSpMV_bufferSize           hipsparseSpMV_bufferSize
#define cusparseCreateDnMat               hipsparseCreateDnMat
#define cusparseSpMV                      hipsparseSpMV
#define cusparseCreateCsc                 hipsparseCreateCsc
#define cusparseSpMM                      hipsparseSpMM
#define cusparseSDDMM_bufferSize          hipsparseSDDMM_bufferSize
#define cusparseSDDMM                     hipsparseSDDMM
#define cusparseSpMM_bufferSize           hipsparseSpMM_bufferSize
#define cusparseSparseToDense             hipsparseSparseToDense
#define cusparseSparseToDense_bufferSize  hipsparseSparseToDense_bufferSize