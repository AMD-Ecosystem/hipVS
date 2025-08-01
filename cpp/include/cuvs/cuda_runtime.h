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

#include <hip/hip_runtime.h>

// types
#define cudaDataType_t              hipDataType
#define cudaFuncAttributes          hipFuncAttributes
#define cudaDeviceProp              hipDeviceProp_t
#define cudaError_t                 hipError_t
#define cudaEvent_t                 hipEvent_t
#define cudaMemAllocationHandleType hipMemAllocationHandleType
#define cudaMemPool_t               hipMemPool_t
#define cudaMemPoolAttr             hipMemPoolAttr
#define cudaMemPoolProps            hipMemPoolProps
#define cudaPointerAttributes       hipPointerAttribute_t
#define cudaStream_t                hipStream_t

// macros, enum constant definitions
#define cudaDevAttrComputeCapabilityMajor              hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor              hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrHostRegisterReadOnlySupported       hipDeviceAttributeHostRegisterReadOnlySupported
#define cudaDevAttrL2CacheSize                         hipDeviceAttributeL2CacheSize
#define cudaDevAttrMaxSharedMemoryPerBlock             hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxThreadsPerBlock                  hipDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrMemoryPoolsSupported                hipDeviceAttributeMemoryPoolsSupported
#define cudaDevAttrMemoryPoolSupportedHandleTypes      hipDevAttrMemoryPoolSupportedHandleTypes
#define cudaDevAttrMultiProcessorCount                 hipDeviceAttributeMultiprocessorCount
#define cudaErrorInvalidValue                          hipErrorInvalidValue
#define cudaErrorMemoryAllocation                      hipErrorMemoryAllocation
#define cudaErrorNotReady                              hipErrorNotReady
#define cudaEventDisableTiming                         hipEventDisableTiming
#define cudaFuncAttributeMaxDynamicSharedMemorySize    hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncCachePreferL1                          hipFuncCachePreferL1
#define cudaFuncCachePreferShared                      hipFuncCachePreferShared
#define cudaHostRegisterMapped                         hipHostRegisterMapped
#define cudaHostRegisterReadOnly                       hipHostRegisterReadOnly
#define cudaMemAllocationTypePinned                    hipMemAllocationTypePinned
#define cudaMemcpyDefault                              hipMemcpyDefault
#define cudaMemcpyDeviceToDevice                       hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost                         hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice                         hipMemcpyHostToDevice
#define cudaMemHandleTypeNone                          hipMemHandleTypeNone
#define cudaMemLocationTypeDevice                      hipMemLocationTypeDevice
#define cudaMemoryTypeDevice                           hipMemoryTypeDevice
#define cudaMemoryTypeHost                             hipMemoryTypeHost
#define cudaMemoryTypeManaged                          hipMemoryTypeManaged
#define cudaMemoryTypeUnregistered                     hipMemoryTypeUnregistered
#define cudaMemPoolAttrReleaseThreshold                hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReleaseThreshold                hipMemPoolAttrReleaseThreshold
#define cudaMemPoolReuseAllowOpportunistic             hipMemPoolReuseAllowOpportunistic
#define cudaMemset                                     hipMemset
#define cudaStreamNonBlocking                          hipStreamNonBlocking
#define cudaStreamPerThread                            hipStreamPerThread
#define cudaSuccess                                    hipSuccess

// functions
#define cudaDeviceGetAttribute      hipDeviceGetAttribute
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaDeviceSynchronize       hipDeviceSynchronize
#define cudaDriverGetVersion        hipDriverGetVersion
#define cudaEventCreate             hipEventCreate
#define cudaEventCreateWithFlags    hipEventCreateWithFlags
#define cudaEventDestroy            hipEventDestroy
#define cudaEventElapsedTime        hipEventElapsedTime
#define cudaEventQuery              hipEventQuery
#define cudaEventRecord             hipEventRecord
#define cudaEventSynchronize        hipEventSynchronize
#define cudaFree                    hipFree
#define cudaFreeAsync               hipFreeAsync
#define cudaFreeHost                hipHostFree
#define cudaFuncGetAttributes       hipFuncGetAttributes
#define cudaFuncSetCacheConfig      hipFuncSetCacheConfig
#define cudaFuncSetAttribute(function, attr, value) \
  hipFuncSetAttribute(reinterpret_cast<const void*>(function), attr, value)
#define cudaGetDevice                                  hipGetDevice
#define cudaGetDeviceCount                             hipGetDeviceCount
#define cudaGetDeviceProperties                        hipGetDeviceProperties
#define cudaGetErrorName                               hipGetErrorName
#define cudaGetErrorString                             hipGetErrorString
#define cudaGetLastError                               hipGetLastError
#define cudaHostGetDevicePointer                       hipHostGetDevicePointer
#define cudaHostUnregister                             hipHostUnregister
#define cudaLaunchKernel                               hipLaunchKernel
#define cudaMalloc                                     hipMalloc
#define cudaMallocAsync                                hipMallocAsync
#define cudaMallocFromPoolAsync                        hipMallocFromPoolAsync
#define cudaMallocHost                                 hipHostMalloc
#define cudaMallocManaged                              hipMallocManaged
#define cudaMemcpy                                     hipMemcpy
#define cudaMemcpy2DAsync                              hipMemcpy2DAsync
#define cudaMemcpyAsync                                hipMemcpyAsync
#define cudaMemGetInfo                                 hipMemGetInfo
#define cudaMemPoolCreate                              hipMemPoolCreate
#define cudaMemPoolDestroy                             hipMemPoolDestroy
#define cudaMemPoolSetAttribute                        hipMemPoolSetAttribute
#define cudaMemsetAsync                                hipMemsetAsync
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize             hipOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSizeVariableSMem hipOccupancyMaxPotentialBlockSizeVariableSMem
#define cudaPeekAtLastError                            hipPeekAtLastError
#define cudaPointerGetAttributes                       hipPointerGetAttributes
#define cudaSetDevice                                  hipSetDevice
#define cudaStreamCreate                               hipStreamCreate
#define cudaStreamCreateWithFlags                      hipStreamCreateWithFlags
#define cudaStreamDestroy                              hipStreamDestroy
#define cudaStreamSynchronize                          hipStreamSynchronize
#define cudaStreamWaitEvent(a, b, c)                   hipStreamWaitEvent(a, b, c)
#define cudaLaunchCooperativeKernel                    hipLaunchCooperativeKernel
