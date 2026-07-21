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

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/integer_utils.hpp>

#include "ivf_common.cuh"

#include <atomic>
#include <fstream>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuvs::neighbors::ivf {

template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
list<SpecT, SizeT, SpecExtraArgs...>::list(raft::resources const& res,
                                           const spec_type& spec,
                                           size_type n_rows)
  : size{n_rows}, data{res}, indices{res}
{
  const SizeT warp_size = raft::host_warp_size(raft::resource::get_device_id(res));
  // On wave32 (RDNA), host_warp_size()=32. Some index types (e.g. IVF-SQ) interleave data in
  // groups of kIndexGroupSize=64 and require device lists to be at least 64-row aligned.
  // Use spec.align_min when present to enforce the minimum alignment regardless of warp size.
  SizeT min_align = warp_size;
  if constexpr (requires { spec.align_min; }) {
    min_align = std::max(min_align, static_cast<SizeT>(spec.align_min));
  }
  const SizeT align_max = spec.conservative_memory_allocation ? min_align : SizeT{1024};
  SizeT capacity        = raft::round_up_safe<SizeT>(n_rows, align_max);
  if (n_rows < align_max) {
    capacity = raft::bound_by_power_of_two<SizeT>(std::max<SizeT>(n_rows, min_align));
    capacity = std::min<SizeT>(capacity, align_max);
  }
  try {
    data    = raft::make_device_mdarray<value_type>(res, spec.make_list_extents(capacity));
    indices = raft::make_device_vector<index_type, SizeT>(res, capacity);
  } catch (std::bad_alloc& e) {
    RAFT_FAIL(
      "ivf::list: failed to allocate a big enough list to hold all data "
      "(requested size: %zu records, selected capacity: %zu records). "
      "Allocator exception: %s",
      size_t(size),
      size_t(capacity),
      e.what());
  }
  // Fill the index buffer with a pre-defined marker for easier debugging
  raft::matrix::fill(res, indices.view(), ivf::kInvalidRecord<index_type>);
}

template <typename ListT>
void resize_list(raft::resources const& res,
                 std::shared_ptr<ListT>& orig_list,  // NOLINT
                 const typename ListT::spec_type& spec,
                 typename ListT::size_type new_used_size,
                 typename ListT::size_type old_used_size)
{
  bool skip_resize = false;
  if (orig_list) {
    if (new_used_size <= orig_list->indices.extent(0)) {
      auto shared_list_size = old_used_size;
      if (new_used_size <= old_used_size ||
          orig_list->size.compare_exchange_strong(shared_list_size, new_used_size)) {
        // We don't need to resize the list if:
        //  1. The list exists
        //  2. The new size fits in the list
        //  3. The list doesn't grow or no-one else has grown it yet
        skip_resize = true;
      }
    }
  } else {
    old_used_size = 0;
  }
  if (skip_resize) { return; }
  auto new_list = std::make_shared<ListT>(res, spec, new_used_size);
  if (old_used_size > 0) {
    auto copied_data_extents = spec.make_list_extents(old_used_size);
    auto copied_view         = raft::make_mdspan<typename ListT::value_type,
                                                 typename ListT::size_type,
                                                 raft::row_major,
                                                 false,
                                                 true>(new_list->data.data_handle(), copied_data_extents);
    raft::copy(res,
               raft::make_device_vector_view(copied_view.data_handle(), copied_view.size()),
               raft::make_device_vector_view(orig_list->data.data_handle(), copied_view.size()));
    raft::copy(res,
               raft::make_device_vector_view(new_list->indices.data_handle(), old_used_size),
               raft::make_device_vector_view(orig_list->indices.data_handle(), old_used_size));
  }
  // swap the shared pointer content with the new list
  new_list.swap(orig_list);
}

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(const raft::resources& handle,
                                             std::ostream& os,
                                             const ListT& ld,
                                             const typename ListT::spec_type& store_spec,
                                             std::optional<typename ListT::size_type> size_override)
{
  using size_type = typename ListT::size_type;
  auto size       = size_override.value_or(ld.size.load());
  raft::serialize_scalar(handle, os, size);
  if (size == 0) { return; }

  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    raft::make_host_mdarray<typename ListT::value_type, size_type, raft::row_major>(data_extents);
  auto inds_array = raft::make_host_mdarray<typename ListT::index_type, size_type, raft::row_major>(
    raft::make_extents<size_type>(size));
  raft::copy(handle,
             raft::make_host_vector_view(data_array.data_handle(), data_array.size()),
             raft::make_device_vector_view(ld.data.data_handle(), data_array.size()));
  raft::copy(handle,
             raft::make_host_vector_view(inds_array.data_handle(), inds_array.size()),
             raft::make_device_vector_view(ld.indices.data_handle(), inds_array.size()));
  raft::resource::sync_stream(handle);
  raft::serialize_mdspan(handle, os, data_array.view());
  raft::serialize_mdspan(handle, os, inds_array.view());
}

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(const raft::resources& handle,
                                             std::ostream& os,
                                             const std::shared_ptr<ListT>& ld,
                                             const typename ListT::spec_type& store_spec,
                                             std::optional<typename ListT::size_type> size_override)
{
  if (ld) {
    return serialize_list<ListT>(handle, os, *ld, store_spec, size_override);
  } else {
    return raft::serialize_scalar(handle, os, typename ListT::size_type{0});
  }
}

template <typename ListT>
enable_if_valid_list_t<ListT> deserialize_list(const raft::resources& handle,
                                               std::istream& is,
                                               std::shared_ptr<ListT>& ld,
                                               const typename ListT::spec_type& store_spec,
                                               const typename ListT::spec_type& device_spec)
{
  using size_type = typename ListT::size_type;
  auto size       = raft::deserialize_scalar<size_type>(handle, is);
  if (size == 0) { return ld.reset(); }
  std::make_shared<ListT>(handle, device_spec, size).swap(ld);
  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    raft::make_host_mdarray<typename ListT::value_type, size_type, raft::row_major>(data_extents);
  auto inds_array = raft::make_host_mdarray<typename ListT::index_type, size_type, raft::row_major>(
    raft::make_extents<size_type>(size));
  raft::deserialize_mdspan(handle, is, data_array.view());
  raft::deserialize_mdspan(handle, is, inds_array.view());
  raft::copy(handle,
             raft::make_device_vector_view(ld->data.data_handle(), data_array.size()),
             raft::make_host_vector_view(data_array.data_handle(), data_array.size()));
  // NB: copying exactly 'size' indices to leave the rest 'kInvalidRecord' intact.
  raft::copy(handle,
             raft::make_device_vector_view(ld->indices.data_handle(), size),
             raft::make_host_vector_view(inds_array.data_handle(), size));
  // Make sure the data is copied from host to device before the host arrays get out of the scope.
  raft::resource::sync_stream(handle);
}
}  // namespace cuvs::neighbors::ivf
