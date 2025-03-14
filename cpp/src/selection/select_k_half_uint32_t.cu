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

#include "./select_k.cuh"

// TODO: (HIP/AMD) Requires resolution of https://github.com/ROCm/clr/issues/147 for this specialization to be enabled.
// ============================================================================================
//  Exact error: message:
//   hipVS/cpp/build/_deps/raft-src/cpp/include/raft/util/cudart_utils.hpp:476:75: note: assignment
//   to member '__x' of union with active member 'data' is not allowed in a constant expression 476
//   |   constexpr explicit inline __half_constexpr(uint16_t u) : __half() { __x = u; }
// ============================================================================================
// The default construction of the __half type is not marked constexpr.
//
// instantiate_cuvs_selection_select_k(half, uint32_t);
