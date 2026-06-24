# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# MIT License
#
# Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import numpy as np
import pytest
from cuda.bindings.runtime import cudaDeviceProp, cudaGetDeviceProperties
from pylibraft.common import device_ndarray

from cuvs.neighbors import vamana


def get_device_warp_size():
    try:
        prop = cudaDeviceProp()
        cudaGetDeviceProperties(prop, 0)
    except TypeError:
        (_, prop) = cudaGetDeviceProperties(0)
    return prop.warpSize


# Initialize graph_degree and visited_size based on device warp size
graph_degree = get_device_warp_size()
visited_size = graph_degree * 2


def _gen_data(shape, dtype):
    rng = np.random.default_rng(12345)
    if dtype == np.float32:
        return rng.random(shape, dtype=np.float32)
    if dtype == np.int8:
        # keep small magnitude to avoid overflow if used elsewhere
        return rng.integers(low=-10, high=10, size=shape, dtype=np.int8)
    if dtype == np.uint8:
        return rng.integers(low=0, high=20, size=shape, dtype=np.uint8)
    raise AssertionError("unexpected dtype in test helper")


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
def test_vamana_build_basic(dtype):
    n_rows, n_cols = 1000, 16
    data = _gen_data((n_rows, n_cols), dtype)
    data_dev = device_ndarray(data)

    params = vamana.IndexParams(
        metric="sqeuclidean",
        graph_degree=graph_degree,
        visited_size=visited_size,
    )
    idx = vamana.build(params, data_dev)

    # Basic sanity: object type and flags
    assert isinstance(idx, vamana.Index)
    assert idx.trained is True
    # repr should include trained flag
    assert "Index(type=Vamana" in repr(idx)


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
@pytest.mark.skip(
    reason="Skipping host build test because of CUDA error "
    "in C++ API. Reference issue: "
    "https://github.com/rapidsai/cuvs/issues/1380"
)
def test_vamana_build_basic_host(dtype):
    n_rows, n_cols = 512, 12
    data = _gen_data((n_rows, n_cols), dtype)  # host array

    params = vamana.IndexParams(
        metric="sqeuclidean",
        graph_degree=graph_degree,
        visited_size=visited_size,
    )
    idx = vamana.build(params, data)

    assert isinstance(idx, vamana.Index)
    assert idx.trained is True


@pytest.mark.parametrize("include_dataset", [True, False])
def test_vamana_serialize(tmp_path, include_dataset):
    n_rows, n_cols = 256, 8
    data = _gen_data((n_rows, n_cols), np.float32)
    data_dev = device_ndarray(data)

    params = vamana.IndexParams(
        graph_degree=graph_degree, visited_size=visited_size
    )
    idx = vamana.build(params, data_dev)

    out_path = tmp_path / "vamana_index.bin"
    vamana.save(str(out_path), idx, include_dataset=include_dataset)

    assert out_path.exists() and out_path.stat().st_size > 0


def test_vamana_build_rejects_unsupported_dtype():
    # float16 is not in the accepted list in the build wrapper; expect failure
    n_rows, n_cols = 64, 8
    data = _gen_data((n_rows, n_cols), np.float32).astype(np.float16)
    from pylibraft.common import device_ndarray

    data_dev = device_ndarray(data)
    params = vamana.IndexParams(
        graph_degree=graph_degree, visited_size=visited_size
    )

    with pytest.raises(Exception):
        _ = vamana.build(params, data_dev)
