/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/nn_descent.h>
#include <stdint.h>

void run_all_neighbors(int64_t n_rows,
                       int64_t n_dim,
                       int64_t k,
                       float* dataset_data,
                       int64_t* indices_data,
                       float* distances_data,
                       float* core_distances_data,
                       cuvsDistanceType metric,
                       cuvsAllNeighborsAlgo algo,
                       float alpha)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create all neighbors index parameters
  cuvsAllNeighborsIndexParams_t params;
  cuvsAllNeighborsIndexParamsCreate(&params);

  // configure parameters
  params->algo           = algo;
  params->metric         = metric;
  params->overlap_factor = 1;
  params->n_clusters     = 1;

  // For NN Descent, configure quality parameters to match the C++ test expectations
  cuvsNNDescentIndexParams_t nnd_params = NULL;
  if (algo == CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT) {
    cuvsNNDescentIndexParamsCreate(&nnd_params);
    nnd_params->graph_degree              = (size_t)k;
    nnd_params->intermediate_graph_degree = (size_t)(k * 2);
    nnd_params->max_iterations            = 100;
    params->nn_descent_params             = nnd_params;
  }

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.device.device_id   = 0;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;
  dataset_tensor.dl_tensor.byte_offset        = 0;
  dataset_tensor.manager_ctx                  = NULL;
  dataset_tensor.deleter                      = NULL;

  // create indices DLTensor
  DLManagedTensor indices_tensor;
  indices_tensor.dl_tensor.data               = (void*)indices_data;
  indices_tensor.dl_tensor.device.device_type = kDLCUDA;
  indices_tensor.dl_tensor.device.device_id   = 0;
  indices_tensor.dl_tensor.ndim               = 2;
  indices_tensor.dl_tensor.dtype.code         = kDLInt;
  indices_tensor.dl_tensor.dtype.bits         = 64;
  indices_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t indices_shape[2]                    = {n_rows, k};
  indices_tensor.dl_tensor.shape              = indices_shape;
  indices_tensor.dl_tensor.strides            = NULL;
  indices_tensor.dl_tensor.byte_offset        = 0;
  indices_tensor.manager_ctx                  = NULL;
  indices_tensor.deleter                      = NULL;

  // create distances DLTensor (optional)
  DLManagedTensor* distances_ptr = NULL;
  DLManagedTensor distances_tensor;
  int64_t distances_shape[2] = {n_rows, k};  // Moved outside if block
  if (distances_data != NULL) {
    distances_tensor.dl_tensor.data               = (void*)distances_data;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id   = 0;
    distances_tensor.dl_tensor.ndim               = 2;
    distances_tensor.dl_tensor.dtype.code         = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits         = 32;
    distances_tensor.dl_tensor.dtype.lanes        = 1;
    distances_tensor.dl_tensor.shape              = distances_shape;
    distances_tensor.dl_tensor.strides            = NULL;
    distances_tensor.dl_tensor.byte_offset        = 0;
    distances_tensor.manager_ctx                  = NULL;
    distances_tensor.deleter                      = NULL;
    distances_ptr                                 = &distances_tensor;
  }

  // create core_distances DLTensor (optional)
  DLManagedTensor* core_distances_ptr = NULL;
  DLManagedTensor core_distances_tensor;
  int64_t core_distances_shape[1] = {n_rows};  // Moved outside if block
  if (core_distances_data != NULL) {
    core_distances_tensor.dl_tensor.data               = (void*)core_distances_data;
    core_distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    core_distances_tensor.dl_tensor.device.device_id   = 0;
    core_distances_tensor.dl_tensor.ndim               = 1;
    core_distances_tensor.dl_tensor.dtype.code         = kDLFloat;
    core_distances_tensor.dl_tensor.dtype.bits         = 32;
    core_distances_tensor.dl_tensor.dtype.lanes        = 1;
    core_distances_tensor.dl_tensor.shape              = core_distances_shape;
    core_distances_tensor.dl_tensor.strides            = NULL;
    core_distances_tensor.dl_tensor.byte_offset        = 0;
    core_distances_tensor.manager_ctx                  = NULL;
    core_distances_tensor.deleter                      = NULL;
    core_distances_ptr                                 = &core_distances_tensor;
  }

  // build all neighbors graph
  cuvsError_t result = cuvsAllNeighborsBuild(
    res, params, &dataset_tensor, &indices_tensor, distances_ptr, core_distances_ptr, alpha);

  // cleanup
  cuvsAllNeighborsIndexParamsDestroy(params);
  cuvsResourcesDestroy(res);
}
