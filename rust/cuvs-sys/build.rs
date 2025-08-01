/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

use std::env;
use std::io::BufRead;
use std::path::PathBuf;
use bindgen::callbacks::{EnumVariantValue, ParseCallbacks};

/// HipToCuda implements the ParseCallbacks trait to rename all the hip* types to cuda* types
/// in the generated bindings. This is needed because the rust code is written to use the CUDA
/// runtime API's, but bindgen(which uses libclang) on the other hand is generating bindings
/// for the HIP version as it sees past the preprocessor defines that convert the CUDA API to
/// the HIP API in our C++ code.
#[derive(Debug)]
struct HipToCuda;

/// Helper: change a leading "hip" to "cuda"
fn rename(original: &str) -> Option<String> {
    original.strip_prefix("hip").map(|tail| format!("cuda{tail}"))
}

impl ParseCallbacks for HipToCuda {
    /// Rename types, functions, globals, etc.
    fn item_name(&self, original: bindgen::callbacks::ItemInfo) -> Option<String> {
        rename(original.name)
    }

    /// Rename individual enum *variants* (needed for hipSuccess → cudaSuccess …).
    fn enum_variant_name(
        &self,
        _enum_name: Option<&str>,
        original: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        rename(original)
    }
}

fn main() {
    // build the cuvs c-api library with cmake, and link it into this crate
    let cuvs_build = cmake::Config::new(".")
        .build();

    println!(
        "cargo:rustc-link-search=native={}/lib",
        cuvs_build.display()
    );
    println!(
        "cargo:rustc-link-search=native=/opt/rocm/lib" // Assume that ROCm is installed in /opt/rocm
    );
    println!("cargo:rustc-link-lib=dylib=cuvs_c") ;
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    // we need some extra flags both to link against cuvs, and also to run bindgen
    // specifically we need to:
    //  * -I flags to set the include path to pick up cudaruntime.h during bindgen
    //  * -rpath-link settings to link to libraft/libcuvs.so etc during the link
    // Rather than redefine the logic to set all these things, lets pick up the values from
    // the cuvs cmake build in its CMakeCache.txt and set from there
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cmake_cache: Vec<String> = std::io::BufReader::new(
        std::fs::File::open(format!("{}/build/CMakeCache.txt", out_path.display()))
            .expect("Failed to open cuvs CMakeCache.txt"),
    )
    .lines()
    .map(|x| x.expect("Couldn't parse line from CMakeCache.txt"))
    .collect();
    let cuvs_c_lib_path = PathBuf::from(cmake_cache
        .iter()
        .find(|x| x.starts_with("CUVS_C_LIBRARY_SO_PATH:FILEPATH="))
        .expect("failed to find CUVS_C_LIBRARY_SO_PATH in CMakeCache.txt")
        .strip_prefix("CUVS_C_LIBRARY_SO_PATH:FILEPATH=")
        .unwrap());
    let cuvs_lib_dir = cuvs_c_lib_path.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", cuvs_lib_dir.display());
    let cmake_cxx_flags = cmake_cache
        .iter()
        .find(|x| x.starts_with("CMAKE_CXX_FLAGS:STRING="))
        .expect("failed to find CMAKE_CXX_FLAGS in CMakeCache.txt")
        .strip_prefix("CMAKE_CXX_FLAGS:STRING=")
        .unwrap();

    let cmake_linker_flags = cmake_cache
        .iter()
        .find(|x| x.starts_with("CMAKE_EXE_LINKER_FLAGS:STRING="))
        .expect("failed to find CMAKE_EXE_LINKER_FLAGS in CMakeCache.txt")
        .strip_prefix("CMAKE_EXE_LINKER_FLAGS:STRING=")
        .unwrap();

    // need to propagate the rpath-link settings to dependent crates =(
    // (this will get added as DEP_CUVS_CMAKE_LINKER_ARGS in dependent crates)
    println!("cargo:cmake_linker_flags={}", cmake_linker_flags);

    // add the required rpath-link flags to the cargo build
    for flag in cmake_linker_flags.split(' ') {
        if flag.starts_with("-Wl,-rpath-link") {
            println!("cargo:rustc-link-arg={}", flag);
        }
    }

    // run bindgen to automatically create rust bindings for the cuvs c-api
    bindgen::Builder::default()
        .header("cuvs_c_wrapper.h")
        .clang_arg("-I../../cpp/include")
        .clang_arg("-D__HIP_PLATFORM_AMD__")// Usually set by the cmake build or hipcc
        .clang_arg("-D__HIP_ROCclr__=1")// Usually set by the cmake build or hipcc
        .clang_arg("-I/opt/rocm/include") // Assume that ROCm is installed in /opt/rocm
        // needed to find cudaruntime.h
        .clang_args(cmake_cxx_flags.split(' '))
        // include dlpack from the cmake build dependencies
        .clang_arg(format!(
            "-I{}/build/_deps/dlpack-src/include/",
            out_path.display()
        ))
        // add `must_use' declarations to functions returning cuvsError_t
        // (so that if you don't check the error code a compile warning is
        // generated)
        .must_use_type("cuvsError_t")
        // Only generate bindings for cuvs/cagra types and functions
        .allowlist_type("(cuvs|bruteForce|cagra|DL).*")
        .allowlist_function("(cuvs|bruteForce|cagra).*")
        .rustified_enum("(cuvs|cagra|DL|DistanceType|codebook_gen|cudaDataType_t).*")
        // also need some basic cuda mem functions for copying data
        .allowlist_function("(hipMemcpyAsync|hipMemcpy)")
        .rustified_enum("hipError_t")
        .parse_callbacks(Box::new(HipToCuda))
        .generate()
        .expect("Unable to generate cagra_c bindings")
        .write_to_file(out_path.join("cuvs_bindings.rs"))
        .expect("Failed to write generated rust bindings");
}
