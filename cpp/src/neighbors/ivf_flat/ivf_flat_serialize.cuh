/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include "../ivf_common.cuh"
#include "../ivf_list.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/pow2_utils.cuh>

#include <fstream>

namespace cuvs::neighbors::ivf_flat::detail {

// Serialization version
// No backward compatibility yet; that is, can't add additional fields without breaking
// backward compatibility.
// TODO(hcho3) Implement next-gen serializer for IVF that allows for expansion in a backward
//             compatible fashion.
constexpr int serialization_version = 4;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index_ IVF-Flat index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& handle, std::ostream& os, const index<T, IdxT>& index_)
{
  RAFT_LOG_DEBUG(
    "Saving IVF-Flat index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  std::string dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  serialize_scalar(handle, os, serialization_version);
  serialize_scalar(handle, os, index_.size());
  serialize_scalar(handle, os, index_.dim());
  serialize_scalar(handle, os, index_.n_lists());
  serialize_scalar(handle, os, index_.metric());
  serialize_scalar(handle, os, index_.adaptive_centers());
  serialize_scalar(handle, os, index_.conservative_memory_allocation());
  serialize_mdspan(handle, os, index_.centers());
  if (index_.center_norms()) {
    bool has_norms = true;
    serialize_scalar(handle, os, has_norms);
    serialize_mdspan(handle, os, *index_.center_norms());
  } else {
    bool has_norms = false;
    serialize_scalar(handle, os, has_norms);
  }
  auto sizes_host = raft::make_host_vector<uint32_t, uint32_t>(index_.list_sizes().extent(0));
  raft::copy(handle, sizes_host.view(), index_.list_sizes());
  raft::resource::sync_stream(handle);
  serialize_mdspan(handle, os, sizes_host.view());

  list_spec<uint32_t, T, IdxT> list_store_spec{index_.dim(), true};
  const uint32_t warp_size = raft::host_warp_size(raft::resource::get_device_id(handle));

  if (warp_size == 32) {
    for (uint32_t label = 0; label < index_.n_lists(); label++) {
      ivf::serialize_list(handle,
                          os,
                          index_.lists()[label],
                          list_store_spec,
                          raft::Pow2<32>::roundUp(sizes_host(label)));
    }
  } else if (warp_size == 64) {
    for (uint32_t label = 0; label < index_.n_lists(); label++) {
      ivf::serialize_list(handle,
                          os,
                          index_.lists()[label],
                          list_store_spec,
                          raft::Pow2<64>::roundUp(sizes_host(label)));
    }
  }
  raft::resource::sync_stream(handle);
}

template <typename T, typename IdxT>
void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<T, IdxT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(handle, of, index_);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

/** Load an index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index_ IVF-Flat index
 *
 */
template <typename T, typename IdxT>
auto deserialize(raft::resources const& handle, std::istream& is) -> index<T, IdxT>
{
  char dtype_string[4];
  is.read(dtype_string, 4);

  auto ver = raft::deserialize_scalar<int>(handle, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows           = raft::deserialize_scalar<IdxT>(handle, is);
  auto dim              = raft::deserialize_scalar<std::uint32_t>(handle, is);
  auto n_lists          = raft::deserialize_scalar<std::uint32_t>(handle, is);
  auto metric           = raft::deserialize_scalar<cuvs::distance::DistanceType>(handle, is);
  bool adaptive_centers = raft::deserialize_scalar<bool>(handle, is);
  bool cma              = raft::deserialize_scalar<bool>(handle, is);

  index<T, IdxT> index_ = index<T, IdxT>(handle, metric, n_lists, adaptive_centers, cma, dim);

  deserialize_mdspan(handle, is, index_.centers());
  bool has_norms = raft::deserialize_scalar<bool>(handle, is);
  if (has_norms) {
    index_.allocate_center_norms(handle);
    if (!index_.center_norms()) {
      RAFT_FAIL("Error inconsistent center norms");
    } else {
      auto center_norms = index_.center_norms().value();
      deserialize_mdspan(handle, is, center_norms);
    }
  }
  deserialize_mdspan(handle, is, index_.list_sizes());

  list_spec<uint32_t, T, IdxT> list_device_spec{index_.dim(), cma};
  list_spec<uint32_t, T, IdxT> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::deserialize_list(handle, is, index_.lists()[label], list_store_spec, list_device_spec);
  }
  raft::resource::sync_stream(handle);

  ivf::detail::recompute_internal_state(handle, index_);

  return index_;
}

template <typename T, typename IdxT>
auto deserialize(raft::resources const& handle, const std::string& filename) -> index<T, IdxT>
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);

  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto index = detail::deserialize<T, IdxT>(handle, is);

  is.close();

  return index;
}
}  // namespace cuvs::neighbors::ivf_flat::detail

#define CUVS_INST_IVF_FLAT_SERIALIZE(T, IdxT)                                           \
  void serialize(raft::resources const& handle,                                         \
                 const std::string& filename,                                           \
                 const cuvs::neighbors::ivf_flat::index<T, IdxT>& index)                \
  {                                                                                     \
    cuvs::neighbors::ivf_flat::detail::serialize(handle, filename, index);              \
  }                                                                                     \
                                                                                        \
  void serialize(raft::resources const& handle,                                         \
                 std::ostream& os,                                                      \
                 const cuvs::neighbors::ivf_flat::index<T, IdxT>& index)                \
  {                                                                                     \
    cuvs::neighbors::ivf_flat::detail::serialize(handle, os, index);                    \
  }                                                                                     \
                                                                                        \
  void deserialize(raft::resources const& handle,                                       \
                   const std::string& filename,                                         \
                   cuvs::neighbors::ivf_flat::index<T, IdxT>* index)                    \
  {                                                                                     \
    *index = cuvs::neighbors::ivf_flat::detail::deserialize<T, IdxT>(handle, filename); \
  }                                                                                     \
  void deserialize(raft::resources const& handle,                                       \
                   std::istream& is,                                                    \
                   cuvs::neighbors::ivf_flat::index<T, IdxT>* index)                    \
  {                                                                                     \
    *index = cuvs::neighbors::ivf_flat::detail::deserialize<T, IdxT>(handle, is);       \
  }
