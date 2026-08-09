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
#pragma once

#include <hip/hiprtc.h>

// types
#ifndef nvrtcProgram
#define nvrtcProgram hiprtcProgram
#endif
#ifndef nvrtcResult
#define nvrtcResult hiprtcResult
#endif

// enum constants
#ifndef NVRTC_SUCCESS
#define NVRTC_SUCCESS HIPRTC_SUCCESS
#endif

// functions
#ifndef nvrtcGetErrorString
#define nvrtcGetErrorString hiprtcGetErrorString
#endif
#ifndef nvrtcCreateProgram
#define nvrtcCreateProgram hiprtcCreateProgram
#endif
#ifndef nvrtcCompileProgram
#define nvrtcCompileProgram hiprtcCompileProgram
#endif
#ifndef nvrtcGetProgramLogSize
#define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#endif
#ifndef nvrtcGetProgramLog
#define nvrtcGetProgramLog hiprtcGetProgramLog
#endif
#ifndef nvrtcDestroyProgram
#define nvrtcDestroyProgram hiprtcDestroyProgram
#endif
// LTO-IR getters — CUDA's NVRTC emits LTO-IR while hipRTC emits LLVM bitcode.
// Both are the compiled device-IR blob handed to the JIT linker's AddData, and
// the two API pairs have identical signatures, so they alias 1-1 here.
#ifndef nvrtcGetLTOIRSize
#define nvrtcGetLTOIRSize hiprtcGetBitcodeSize
#endif
#ifndef nvrtcGetLTOIR
#define nvrtcGetLTOIR hiprtcGetBitcode
#endif
