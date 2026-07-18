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

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use cmake_package::{Error as CmakeError, VersionError, find_package};

const CUVS_COMPONENT: &str = "c_api";
const CUVS_C_API_TARGET: &str = "cuvs::c_api";
const HIPVS_CMAKE_INSPECTION_FAILED: &str = "CMake failed while inspecting hipVS. Check the build environment for missing tools such as ninja/make, C/C++ compilers, or ROCm/HIP dependencies.";
const PYTHON_PRINT_LIBHIPVS_PACKAGE_DIR: &str = r#"
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("libhipvs")
if spec is None or spec.submodule_search_locations is None:
    raise ModuleNotFoundError("libhipvs")

print(Path(next(iter(spec.submodule_search_locations))).resolve())
"#;

pub(crate) struct CuvsMetadata {
    pub(crate) include_dir: PathBuf,
    #[cfg(feature = "generate-bindings")]
    pub(crate) bindgen_include_dirs: Vec<PathBuf>,
    pub(crate) lib_dir: PathBuf,
}

fn cmake_unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "CMake is not installed or does not satisfy this build's requirements. Install the required CMake version and try again."
    )
}

fn hipvs_package_not_found_error() -> anyhow::Error {
    anyhow::anyhow!(
        "Could not find a hipVS (cuVS) CMake package.\n\n\
         Install hipVS via one of:\n\
         - conda: conda install libhipvs\n\
         - pip:   pip install libhipvs and set LIBHIPVS_USE_PYTHON=1\n\
         Or set CMAKE_PREFIX_PATH to point to your hipVS build/install directory."
    )
}

fn hipvs_incompatible_version_error(
    candidates: &[cmake_package::Considered],
) -> anyhow::Error {
    let considered = candidates
        .iter()
        .map(|candidate| format!("- {} (version: {})", candidate.config, candidate.version))
        .collect::<Vec<_>>()
        .join("\n");

    anyhow::anyhow!(
        "Found hipVS (cuVS) CMake package candidates, but none are compatible.\n\n\
         Considered candidates:\n{considered}"
    )
}

fn cmake_package_error(error: CmakeError) -> anyhow::Error {
    match error {
        CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion => {
            cmake_unavailable_error()
        }
        CmakeError::PackageNotFound => hipvs_package_not_found_error(),
        CmakeError::Version(VersionError::VersionIncompatible(candidates)) => {
            hipvs_incompatible_version_error(&candidates)
        }
        CmakeError::Version(VersionError::InvalidVersion) => anyhow::anyhow!(
            "{HIPVS_CMAKE_INSPECTION_FAILED}\n\nCMake reported an invalid package version."
        ),
        CmakeError::IO(error) => {
            anyhow::anyhow!("{HIPVS_CMAKE_INSPECTION_FAILED}\n\nUnderlying error: {error}")
        }
        CmakeError::Internal => anyhow::anyhow!("{HIPVS_CMAKE_INSPECTION_FAILED}"),
    }
}

fn find_target(
    package: &cmake_package::CMakePackage,
    target_name: &str,
) -> Result<cmake_package::CMakeTarget> {
    package.target(target_name).with_context(|| {
        format!("Found CMake package {}, but target {target_name} was not exported.", package.name)
    })
}

fn find_cuvs_package(
    python_package_dir: Option<&Path>,
) -> Result<cmake_package::CMakePackage> {
    let mut package = find_package("cuvs")
        .components([CUVS_COMPONENT.to_owned()])
        .define("CMAKE_FIND_PACKAGE_PREFER_CONFIG", "TRUE");

    if let Some(python_package_dir) = python_package_dir {
        package = package.prefix_paths(vec![python_package_dir.to_path_buf()]);
    }

    package.find().map_err(|error| cmake_package_error(error))
}

#[cfg(feature = "generate-bindings")]
fn find_dlpack_include_dirs() -> Result<Vec<PathBuf>> {
    // Try DLPACK_INCLUDE_DIR env var first.
    if let Ok(dir) = std::env::var("DLPACK_INCLUDE_DIR") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Ok(vec![dir]);
        }
    }

    // Try cmake find_package(dlpack).
    match find_package("dlpack").find() {
        Ok(package) => {
            let target = find_target(&package, "dlpack::dlpack")?;
            return Ok(target
                .include_directories
                .iter()
                .map(PathBuf::from)
                .filter(|dir| dir.is_dir())
                .collect());
        }
        Err(CmakeError::CMakeNotFound | CmakeError::UnsupportedCMakeVersion) => {
            return Err(cmake_unavailable_error());
        }
        Err(_) => {}
    }

    // Fall back to the cuVS include directory (dlpack headers are often
    // co-installed or vendored alongside the cuVS headers).
    Err(anyhow::anyhow!(
        "Could not find DLPack headers.\n\n\
         Set DLPACK_INCLUDE_DIR to point to a directory containing dlpack/dlpack.h,\n\
         or install DLPack so that `find_package(dlpack)` succeeds."
    ))
}

/// Run CMake `find_package(cuvs)` and extract the include and library directories.
/// Calls `CMakeTarget::link()` to emit the full set of cargo link directives,
/// preserving all link libraries, directories, and options from the CMake target.
pub(crate) fn try_find_cuvs_package(
    python_package_dir: Option<&Path>,
) -> Result<CuvsMetadata> {
    let package = find_cuvs_package(python_package_dir)?;
    let target = find_target(&package, CUVS_C_API_TARGET)?;

    let include_dir = target
        .include_directories
        .first()
        .map(PathBuf::from)
        .context("hipVS CMake target did not export any include directories")?;

    #[cfg(feature = "generate-bindings")]
    let bindgen_include_dirs: Vec<_> = {
        let dlpack_dirs = find_dlpack_include_dirs()?;
        dlpack_dirs
            .into_iter()
            .filter(|dir| dir != &include_dir)
            .collect()
    };

    let lib_dir = target
        .location
        .as_deref()
        .and_then(|location| Path::new(location).parent())
        .map(Path::to_path_buf)
        .or_else(|| target.link_directories.first().map(PathBuf::from))
        .context("hipVS CMake target did not export a library location or link directory")?;

    target.link();

    Ok(CuvsMetadata {
        include_dir,
        #[cfg(feature = "generate-bindings")]
        bindgen_include_dirs,
        lib_dir,
    })
}

fn find_python_hipvs_package_dir() -> Result<PathBuf> {
    let python =
        Path::new(if std::env::var_os("VIRTUAL_ENV").is_some() { "python" } else { "python3" });
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_PRINT_LIBHIPVS_PACKAGE_DIR)
        .output()
        .with_context(|| format!("LIBHIPVS_USE_PYTHON is set, but failed to run {:?}.", python))?;

    anyhow::ensure!(
        output.status.success(),
        "LIBHIPVS_USE_PYTHON is set, but {:?} could not locate the Python libhipvs package.\n\n\
             Install the libhipvs wheel in that Python environment, or unset LIBHIPVS_USE_PYTHON.\n\n\
             {}",
        python,
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let package_dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let cmake_dir = package_dir.join("lib64/cmake/cuvs");
    anyhow::ensure!(
        cmake_dir.is_dir(),
        "LIBHIPVS_USE_PYTHON is set, but the Python libhipvs package at {} does not contain a cuVS CMake package under {}.",
        package_dir.display(),
        cmake_dir.display(),
    );

    Ok(package_dir)
}

/// Locate hipVS (cuVS) either from standard CMake search paths or, when explicitly
/// requested, from the active Python environment.
pub(crate) fn locate_hipvs() -> Result<CuvsMetadata> {
    let python_package_dir = std::env::var_os("LIBHIPVS_USE_PYTHON")
        .map(|_| find_python_hipvs_package_dir())
        .transpose()?;

    try_find_cuvs_package(python_package_dir.as_deref())
}
