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
#pragma once
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup pairwise_distance_c C pairwise distance
 * @{
 */

/**
 * @brief Compute pairwise distances for two matrices
 *
 *
 * Usage example:
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/distance/pairwise_distance.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor x;
 * DLManagedTensor y;
 * DLManagedTensor dist;
 *
 * cuvsPairwiseDistance(res, &x, &y, &dist, L2SqrtUnexpanded, 2.0);
 * @endcode
 *
 * @param[in] res cuvs resources object for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
CUVS_EXPORT cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
                                 DLManagedTensor* x,
                                 DLManagedTensor* y,
                                 DLManagedTensor* dist,
                                 cuvsDistanceType metric,
                                 float metric_arg);

/** @} */

#ifdef __cplusplus
}
#endif
