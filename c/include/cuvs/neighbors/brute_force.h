/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
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
// clang-format on

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup bruteforce_c_index Bruteforce index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::brute_force::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBruteForceIndex;

typedef cuvsBruteForceIndex* cuvsBruteForceIndex_t;

/**
 * @brief Allocate BRUTEFORCE index
 *
 * @param[in] index cuvsBruteForceIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index);

/**
 * @brief De-allocate BRUTEFORCE index
 *
 * @param[in] index cuvsBruteForceIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_build Bruteforce index build
 * @{
 */
/**
 * @brief Build a BRUTEFORCE index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 64`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        3. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *
 *        Note that `metric_arg` is a `float` for every dataset type. It is read only by the
 *        Minkowski (`LpUnexpanded`) metric, and a `p` that is not exactly representable in
 *        float32 is rounded, so a float64 search returns the exact Lp distances for the rounded
 *        `p` -- about 1e-7 relative away from the requested ones. Neighbor order is unaffected
 *        unless two candidates are that close together.
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create BRUTEFORCE index
 * cuvsBruteForceIndex_t index;
 * cuvsError_t index_create_status = cuvsBruteForceIndexCreate(&index);
 *
 * // Build the BRUTEFORCE Index
 * cuvsError_t build_status = cuvsBruteForceBuild(res, &dataset_tensor, L2Expanded, 0.f, index);
 *
 * // de-allocate `index` and `res`
 * cuvsError_t index_destroy_status = cuvsBruteForceIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[in] metric metric
 * @param[in] metric_arg metric_arg
 * @param[out] index cuvsBruteForceIndex_t Newly built BRUTEFORCE index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
                                DLManagedTensor* dataset,
                                cuvsDistanceType metric,
                                float metric_arg,
                                cuvsBruteForceIndex_t index);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_search Bruteforce index search
 * @{
 */
/**
 * @brief Search a BRUTEFORCE index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the BRUTEFORCE index must have been built
 *        with the same type of `queries`, such that the `code`, `bits` and `lanes` of
 *        `queries.dl_tensor.dtype` all match `index.dtype`. Types for input are:
 *        1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 64`, `32` or `16`
 *        2. `neighbors`: `kDLDataType.code == kDLInt` and `kDLDataType.bits = 64`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 64` for a
 *          float64 index, `kDLDataType.bits = 32` for a float32 or float16 index
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 * DLManagedTensor bitmap;
 *
 * cuvsFilter prefilter{(uintptr_t)&bitmap, BITMAP};
 *
 * // Search the `index` built using `cuvsBruteForceBuild`
 * cuvsError_t search_status = cuvsBruteForceSearch(res, index, &queries, &neighbors, &distances,
 * prefilter);
 *
 * // de-allocate `res`
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index cuvsBruteForceIndex which has been returned by `cuvsBruteForceBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] prefilter cuvsFilter input prefilter that can be used
              to filter queries and neighbors based on the given bitmap.
 */
CUVS_EXPORT cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
                                 cuvsBruteForceIndex_t index,
                                 DLManagedTensor* queries,
                                 DLManagedTensor* neighbors,
                                 DLManagedTensor* distances,
                                 cuvsFilter prefilter);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_serialize BRUTEFORCE C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.c}
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsBruteforceBuild`
 * cuvsBruteForceSerialize(res, "/path/to/index", index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index BRUTEFORCE index
 *
 */
CUVS_EXPORT cuvsError_t cuvsBruteForceSerialize(cuvsResources_t res,
                                    const char* filename,
                                    cuvsBruteForceIndex_t index);

/**
 * Load index from file.
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * The dtype of the loaded index is read from the file: numpy descriptors `f8`, `f4` and `e2`
 * restore a float64, float32 and float16 index respectively, and `index->dtype` is set
 * accordingly.
 *
 * @code{.c}
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Deserialize an index previously built with `cuvsBruteforceBuild`
 * cuvsBruteForceIndex_t index;
 * cuvsBruteForceIndexCreate(&index);
 * cuvsBruteForceDeserialize(res, "/path/to/index", index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index BRUTEFORCE index loaded disk
 */
CUVS_EXPORT cuvsError_t cuvsBruteForceDeserialize(cuvsResources_t res,
                                      const char* filename,
                                      cuvsBruteForceIndex_t index);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif
