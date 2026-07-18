/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

mod cmake;

#[cfg(feature = "generate-bindings")]
use std::path::{Path, PathBuf};

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path, include_dirs: &[PathBuf]) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo"));
    let stub_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("bindgen-stubs");

    let mut builder = bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*")
        .blocklist_item("cuda.*")
        .blocklist_item("CUstream_st")
        .raw_line("use crate::{cudaDataType_t, cudaStream_t};")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    // The stub dir provides a minimal cuda_runtime.h so bindgen can parse
    // the cuVS C headers without a CUDA or ROCm toolkit.
    builder = builder.clang_arg(format!("-I{}", stub_dir.display()));
    builder = builder.clang_arg(format!("-I{}", include_dir.display()));

    for include_dir in include_dirs {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }

    builder
        .generate()
        .expect("bindgen failed to generate cuvs bindings")
        .write_to_file(out_dir.join("cuvs_bindings.rs"))
        .expect("failed to write cuvs_bindings.rs");
}

fn main() {
    println!("cargo::rerun-if-changed=cmake.rs");
    println!("cargo::rerun-if-changed=bindgen-stubs/cuda_runtime.h");
    println!("cargo::rerun-if-env-changed=CMAKE_PREFIX_PATH");
    println!("cargo::rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo::rerun-if-env-changed=LIBHIPVS_USE_PYTHON");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo::rerun-if-env-changed=DLPACK_INCLUDE_DIR");

    if cfg!(feature = "doc-only") {
        return;
    }

    let metadata = match cmake::locate_hipvs() {
        Ok(metadata) => metadata,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    // Expose include path to downstream crates via DEP_HIPVS_INCLUDE.
    println!("cargo::metadata=include={}", metadata.include_dir.display());
    // Expose the directory containing libcuvs_c.so via DEP_HIPVS_LIB.
    println!("cargo::metadata=lib={}", metadata.lib_dir.display());

    // Link against the HIP runtime.
    println!("cargo::rustc-link-search=native=/opt/rocm/lib");
    println!("cargo::rustc-link-lib=dylib=amdhip64");

    #[cfg(feature = "generate-bindings")]
    generate_bindings(&metadata.include_dir, &metadata.bindgen_include_dirs);
}
