// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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
#ifndef cudaDataType_t
#define cudaDataType_t hipDataType
#endif
#ifndef cudaFuncAttributes
#define cudaFuncAttributes hipFuncAttributes
#endif
#ifndef cudaDeviceProp
#define cudaDeviceProp hipDeviceProp_t
#endif
#ifndef cudaError_t
#define cudaError_t hipError_t
#endif
#ifndef cudaEvent_t
#define cudaEvent_t hipEvent_t
#endif
#ifndef cudaMemAllocationHandleType
#define cudaMemAllocationHandleType hipMemAllocationHandleType
#endif
#ifndef cudaMemPool_t
#define cudaMemPool_t hipMemPool_t
#endif
#ifndef cudaMemPoolAttr
#define cudaMemPoolAttr hipMemPoolAttr
#endif
#ifndef cudaMemPoolProps
#define cudaMemPoolProps hipMemPoolProps
#endif
#ifndef cudaPointerAttributes
#define cudaPointerAttributes hipPointerAttribute_t
#endif
#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif

#ifdef __cplusplus
// JIT module types — cudaKernel_t / cudaLibrary_t are CUDA 12.x driver-level
// types that have no direct HIP macro equivalent.  Map them to the hipModule API.
#ifndef cudaKernel_t
using cudaKernel_t = hipFunction_t;
#endif
#ifndef cudaLibrary_t
using cudaLibrary_t = hipModule_t;
#endif
#endif

// FP16 constants — CUDA defines these in cuda_fp16.h; HIP does not provide
// equivalents, so define them from the IEEE 754 half-precision bit patterns.
#ifndef CUDART_MAX_NORMAL_FP16
#define CUDART_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7BFF)
#endif
#ifndef CUDART_MIN_DENORM_FP16
#define CUDART_MIN_DENORM_FP16 __ushort_as_half((unsigned short)0x0001)
#endif

// macros, enum constant definitions
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef cudaDevAttrL2CacheSize
#define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#endif
#ifndef cudaDevAttrMaxThreadsPerBlock
#define cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#endif
#ifndef cudaDevAttrMemoryPoolsSupported
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#endif
#ifndef cudaDevAttrMemoryPoolSupportedHandleTypes
#define cudaDevAttrMemoryPoolSupportedHandleTypes hipDeviceAttributeMemoryPoolSupportedHandleTypes
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif
#ifndef cudaDevAttrClockRate
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#endif
#ifndef cudaDevAttrMemoryClockRate
#define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
#endif
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue hipErrorInvalidValue
#endif
#ifndef cudaErrorMemoryAllocation
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#endif
#ifndef cudaErrorNotReady
#define cudaErrorNotReady hipErrorNotReady
#endif
#ifndef cudaEventDisableTiming
#define cudaEventDisableTiming hipEventDisableTiming
#endif
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#endif
#ifndef cudaFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#endif
#ifndef cudaFuncCachePreferL1
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#endif
#ifndef cudaFuncCachePreferShared
#define cudaFuncCachePreferShared hipFuncCachePreferShared
#endif
#ifndef cudaHostRegister
#define cudaHostRegister hipHostRegister
#endif
#ifndef cudaHostRegisterMapped
#define cudaHostRegisterMapped hipHostRegisterMapped
#endif
#ifndef cudaHostRegisterReadOnly
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#endif
#ifndef cudaHostRegisterDefault
#define cudaHostRegisterDefault hipHostRegisterDefault
#endif
#ifndef cudaHostRegisterIoMemory
#define cudaHostRegisterIoMemory hipHostRegisterIoMemory
#endif
#ifndef cudaErrorNotSupported
#define cudaErrorNotSupported hipErrorNotSupported
#endif
#ifndef cudaMemAdviseSetPreferredLocation
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#endif
#ifndef cudaMemAllocationTypePinned
#define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#endif
#ifndef cudaMemcpyDefault
#define cudaMemcpyDefault hipMemcpyDefault
#endif
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif
#ifndef cudaMemHandleTypeNone
#define cudaMemHandleTypeNone hipMemHandleTypeNone
#endif
#ifndef cudaMemLocation
#define cudaMemLocation hipMemLocation
#endif
#ifndef cudaMemLocationTypeDevice
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#endif
#ifndef cudaMemLocationTypeHost
#define cudaMemLocationTypeHost hipMemLocationTypeHost
#endif
#ifndef cudaMemoryTypeDevice
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#endif
#ifndef cudaMemoryTypeHost
#define cudaMemoryTypeHost hipMemoryTypeHost
#endif
#ifndef cudaMemoryTypeManaged
#define cudaMemoryTypeManaged hipMemoryTypeManaged
#endif
#ifndef cudaMemoryTypeUnregistered
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#endif
#ifndef cudaMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#endif
#ifndef cudaMemPoolReuseAllowOpportunistic
#define cudaMemPoolReuseAllowOpportunistic hipMemPoolReuseAllowOpportunistic
#endif
#ifndef cudaMemset
#define cudaMemset hipMemset
#endif
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif
#ifndef cudaStreamPerThread
#define cudaStreamPerThread hipStreamPerThread
#endif
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif
#ifndef cudaErrorNoDevice
#define cudaErrorNoDevice hipErrorNoDevice
#endif
#ifndef cudaErrorInvalidDevice
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#endif

// functions
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDeviceGetDefaultMemPool
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#endif
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif
#ifndef cudaDriverGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#endif
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#endif
#ifndef cudaEventCreateWithFlags
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#endif
#ifndef cudaEventDestroy
#define cudaEventDestroy hipEventDestroy
#endif
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif
#ifndef cudaEventQuery
#define cudaEventQuery hipEventQuery
#endif
#ifndef cudaEventRecord
#define cudaEventRecord hipEventRecord
#endif
#ifndef cudaEventSynchronize
#define cudaEventSynchronize hipEventSynchronize
#endif
#ifndef cudaFree
#define cudaFree hipFree
#endif
#ifndef cudaFreeAsync
#define cudaFreeAsync hipFreeAsync
#endif
#ifndef cudaFreeHost
#define cudaFreeHost hipHostFree
#endif
#ifndef cudaFuncGetAttributes
#define cudaFuncGetAttributes hipFuncGetAttributes
#endif
#ifndef cudaFuncSetCacheConfig
#define cudaFuncSetCacheConfig hipFuncSetCacheConfig
#endif
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif
#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount hipGetDeviceCount
#endif
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif
#ifndef cudaGetErrorName
#define cudaGetErrorName hipGetErrorName
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#endif
#ifndef cudaHostUnregister
#define cudaHostUnregister hipHostUnregister
#endif
#ifndef cudaLaunchHostFunc
#define cudaLaunchHostFunc hipLaunchHostFunc
#endif
#ifndef cudaLaunchKernel
#define cudaLaunchKernel hipLaunchKernel
#endif
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif
#ifndef cudaMallocAsync
#define cudaMallocAsync hipMallocAsync
#endif
#ifndef cudaMallocFromPoolAsync
#define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#endif
#ifndef cudaMallocHost
#define cudaMallocHost hipHostMalloc
#endif
#ifndef cudaMallocManaged
#define cudaMallocManaged hipMallocManaged
#endif
#ifndef cudaMemAdvise
#define cudaMemAdvise hipMemAdvise_v2
#endif
#ifndef cudaMemcpy
#define cudaMemcpy hipMemcpy
#endif
#ifndef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif
#ifndef cudaMemGetInfo
#define cudaMemGetInfo hipMemGetInfo
#endif
#ifndef cudaMemPoolCreate
#define cudaMemPoolCreate hipMemPoolCreate
#endif
#ifndef cudaMemPoolDestroy
#define cudaMemPoolDestroy hipMemPoolDestroy
#endif
#ifndef cudaMemPoolSetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#endif
#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif
// hipOccupancyMaxActiveBlocksPerMultiprocessor takes `const void*` (host function
// pointer), but JIT-loaded kernels are hipFunction_t (module handle).  Provide a
// type-safe overload set so callers using either kind compile and route correctly.
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#endif
#ifndef cudaOccupancyMaxPotentialBlockSizeVariableSMem
#define cudaOccupancyMaxPotentialBlockSizeVariableSMem hipOccupancyMaxPotentialBlockSizeVariableSMem
#endif
// CUDA 12.x kernel attribute APIs.  On AMD, dynamic shared memory limits don't
// require an opt-in call (full LDS is available by default), so these are no-ops
// when the kernel handle is a hipFunction_t (module-loaded JIT kernel).
#ifdef __cplusplus
#ifndef cudaFuncSetAttribute
inline hipError_t cudaFuncSetAttribute(hipFunction_t, hipFuncAttribute, int) { return hipSuccess; }
inline hipError_t cudaFuncSetAttribute(const void* func, hipFuncAttribute attr, int value)
{
  return hipFuncSetAttribute(func, attr, value);
}
#endif
#ifndef cudaKernelSetAttributeForDevice
inline hipError_t cudaKernelSetAttributeForDevice(hipFunction_t, hipFuncAttribute, int, int)
{
  return hipSuccess;
}
#endif
// hipOccupancyMaxActiveBlocksPerMultiprocessor takes const void* (host function
// pointer).  For JIT-loaded hipFunction_t, route to the hipModule variant.
// Declared after the macro so the name resolves via ADL, not the macro.
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                               hipFunction_t f,
                                                               int blockSize,
                                                               size_t dynSharedMemPerBlk)
{
  return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks, f, blockSize, dynSharedMemPerBlk);
}
#endif
#ifndef cudaPeekAtLastError
#define cudaPeekAtLastError hipPeekAtLastError
#endif
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#endif
#ifndef cudaSetDevice
#define cudaSetDevice hipSetDevice
#endif
#ifndef cudaStreamCreate
#define cudaStreamCreate hipStreamCreate
#endif
#ifndef cudaStreamCreateWithFlags
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#endif
#ifndef cudaStreamDestroy
#define cudaStreamDestroy hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif
#ifndef cudaStreamWaitEvent
#define cudaStreamWaitEvent(a, b, c) hipStreamWaitEvent(a, b, c)
#endif
#ifndef cudaLaunchCooperativeKernel
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#endif
// JIT module/library APIs.  cudaLibrary_t is aliased to hipModule_t above, so the
// library-unload entry point maps 1-1 to hipModuleUnload.  The load path
// (cudaLibraryLoadData vs hipModuleLoadData) and the JIT linker (nvJitLink vs
// hiprtc) have no 1-1 equivalents and are handled explicitly at their call sites.
#ifndef cudaLibraryUnload
#define cudaLibraryUnload hipModuleUnload
#endif
