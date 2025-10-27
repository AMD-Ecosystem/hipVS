/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

# <<<<<<< OLD CODE from 76eb5a2b2 (fdf3862a0) - COMMENTED OUT
# use std::env;
#
# fn main() {
#     // add the required rpath-link flags to the cargo build
#     // TODO: ... this isn't great, there must be a way to propagate this directly without hacks like
#     // this
#     let cmake_linker_flags = env::var("DEP_HIPVS_CMAKE_LINKER_FLAGS").unwrap();
#     for flag in cmake_linker_flags.split(' ') {
#         if flag.starts_with("-Wl,-rpath-link") {
#             println!("cargo:rustc-link-arg={}", flag);
#         }
#     }
# }
# =======
#include <cuvs/core/all.h>

// Please add any extra required header files below for which rust bindings are needed
# >>>>>>> END NEW CODE (fdf3862a0)
