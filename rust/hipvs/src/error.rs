/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

use std::fmt;

#[derive(Debug, Clone)]
pub struct CuvsError {
    code: ffi::cuvsError_t,
    text: String,
}

#[derive(Debug, Clone)]
pub enum Error {
    CudaError(ffi::cudaError_t),
    CuvsError(CuvsError),
    /// The caller passed an argument that could not be forwarded to the C API
    /// (e.g. a filename containing an interior NUL byte or invalid UTF-8).
    InvalidArgument(String),
}

impl std::error::Error for Error {}
impl std::error::Error for CuvsError {}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::CudaError(cuda_error) => write!(f, "cudaError={:?}", cuda_error),
            Error::CuvsError(cuvs_error) => write!(f, "cuvsError={:?}", cuvs_error),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
        }
    }
}

impl fmt::Display for CuvsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}:{}", self.code, self.text)
    }
}

/// Simple wrapper to convert a cuvsError_t into a Result
pub fn check_cuvs(err: ffi::cuvsError_t) -> Result<()> {
    match err {
        ffi::cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => {
            // get a description of the error from cuvs
            let cstr = unsafe {
                let text_ptr = ffi::cuvsGetLastErrorText();
                std::ffi::CStr::from_ptr(text_ptr)
            };
            let text = std::string::String::from_utf8_lossy(cstr.to_bytes()).to_string();

            Err(Error::CuvsError(CuvsError { code: err, text }))
        }
    }
}

pub fn check_cuda(err: ffi::cudaError_t) -> Result<()> {
    match err {
        ffi::cudaError::cudaSuccess => Ok(()),
        _ => Err(Error::CudaError(err)),
    }
}
