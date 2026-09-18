/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/brute_force.h>
#include <stdint.h>

// `dtype_bits` selects the dataset and query element type: 64 for double, 32 for float. The
// distances follow the index, so they are float64 for a float64 index and float32 otherwise.
void run_brute_force(int64_t n_rows,
                     int64_t n_queries,
                     int64_t n_dim,
                     uint32_t n_neighbors,
                     uint8_t dtype_bits,
                     void* index_data,
                     void* query_data,
                     uint32_t* prefilter_data,
                     enum cuvsFilterType prefilter_type,
                     void* distances_data,
                     int64_t* neighbors_data,
                     cuvsDistanceType metric)
{
  uint8_t distance_bits = (dtype_bits == 64) ? 64 : 32;

  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = dtype_bits;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsBruteForceIndex_t index;
  cuvsBruteForceIndexCreate(&index);

  // build index
  cuvsBruteForceBuild(res, &dataset_tensor, metric, 0.0f, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = (void*)query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = dtype_bits;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = (void*)neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = (void*)distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = distance_bits;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  cuvsFilter prefilter;

  DLManagedTensor prefilter_tensor;
  if (prefilter_data == NULL || prefilter_type == NO_FILTER) {
    prefilter.type = NO_FILTER;
    prefilter.addr = (uintptr_t)NULL;
  } else {
    prefilter_tensor.dl_tensor.data               = (void*)prefilter_data;
    prefilter_tensor.dl_tensor.device.device_type = kDLCUDA;
    prefilter_tensor.dl_tensor.ndim               = 1;
    prefilter_tensor.dl_tensor.dtype.code         = kDLUInt;
    prefilter_tensor.dl_tensor.dtype.bits         = 32;
    prefilter_tensor.dl_tensor.dtype.lanes        = 1;

    int64_t prefilter_bits_num = (prefilter_type == BITMAP) ? n_queries * n_rows : n_rows;
    int64_t prefilter_shape[1] = {(prefilter_bits_num + 31) / 32};

    prefilter_tensor.dl_tensor.shape   = prefilter_shape;
    prefilter_tensor.dl_tensor.strides = NULL;

    prefilter.type = prefilter_type;
    prefilter.addr = (uintptr_t)&prefilter_tensor;
  }

  // search index
  cuvsBruteForceSearch(
    res, index, &queries_tensor, &neighbors_tensor, &distances_tensor, prefilter);

  // de-allocate index and res
  cuvsBruteForceIndexDestroy(index);
  cuvsResourcesDestroy(res);
}
