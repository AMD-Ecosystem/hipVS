
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


function(find_and_configure_ck)
    # The fused-distance/pairwise CK backend uses the header-only ck_tile API
    # (resolved via the base include dir provided by every CK target), not the
    # classic ck::tensor_operation::device::DeviceGemm library, so the
    # device_gemm_operations component is not strictly required.
    #
    # Compatibility note: through ROCm 7.2.x the composable_kernel package ships
    # a device_gemm_operations target. As of ROCm 7.13 it no longer does -- the
    # package Config still lists it as a "supported component" but
    # unconditionally include()s a device_gemm_operationsTargets.cmake that is
    # not installed, which hard-fails find_package. Request the component only
    # when the package actually provides it so we stay compatible with both.
    if(DEFINED ENV{ROCM_PATH})
        set(_ck_rocm_path "$ENV{ROCM_PATH}")
    elseif(DEFINED ROCM_PATH)
        set(_ck_rocm_path "${ROCM_PATH}")
    else()
        set(_ck_rocm_path "/opt/rocm")
    endif()

    set(_ck_components device_other_operations utility)
    if(EXISTS
       "${_ck_rocm_path}/lib/cmake/composable_kernel/composable_kerneldevice_gemm_operationsTargets.cmake"
    )
        list(APPEND _ck_components device_gemm_operations)
    endif()

    find_package(
        composable_kernel REQUIRED COMPONENTS ${_ck_components}
                                            PATHS "${_ck_rocm_path}/lib/cmake"
    )
endfunction()

find_and_configure_ck()
