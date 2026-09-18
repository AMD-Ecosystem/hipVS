# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from scipy.spatial.distance import cdist

from cuvs.neighbors import brute_force, filters


def _reference(dataset, queries, metric, metric_arg=2.0):
    if metric == "inner_product":
        return queries @ dataset.T
    kwargs = (
        {"p": float(np.float32(metric_arg))} if metric == "minkowski" else {}
    )
    return cdist(queries, dataset, metric=metric, **kwargs)


def _check_results(distances, neighbors, expected, k, metric):
    actual_distances = distances.copy_to_host()
    actual_neighbors = neighbors.copy_to_host()
    assert actual_distances.dtype == np.float64
    assert actual_neighbors.dtype == np.int64
    order = np.argsort(-expected if metric == "inner_product" else expected)
    expected_neighbors = order[:, :k]
    np.testing.assert_array_equal(actual_neighbors, expected_neighbors)
    np.testing.assert_allclose(
        actual_distances,
        np.take_along_axis(expected, expected_neighbors, axis=1),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "metric",
    [
        "sqeuclidean",
        "euclidean",
        "cosine",
        "inner_product",
        "cityblock",
        "minkowski",
    ],
)
@pytest.mark.parametrize("dataset_order", ["C", "F"])
@pytest.mark.parametrize("query_order", ["C", "F"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("serialize", [False, True])
def test_brute_force_float64(
    metric, dataset_order, query_order, inplace, serialize, tmp_path
):
    rng = np.random.default_rng(42)
    dataset = np.array(rng.random((97, 17)), order=dataset_order)
    queries = np.array(rng.random((12, 17)), order=query_order)
    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)
    k = 7
    metric_arg = 2.1
    index = brute_force.build(dataset_device, metric, metric_arg=metric_arg)
    assert index.dtype == np.dtype("float64")
    if serialize:
        filename = str(tmp_path / "index.bin")
        brute_force.save(filename, index)
        index = brute_force.load(filename)
        assert index.dtype == np.dtype("float64")

    outputs = {}
    if inplace:
        outputs = {
            "distances": device_ndarray.empty((12, k), dtype="float64"),
            "neighbors": device_ndarray.empty((12, k), dtype="int64"),
        }
    expected = _reference(dataset, queries, metric, metric_arg)
    # Repeat to verify that supplied output buffers can be reused.
    for _ in range(2 if inplace else 1):
        distances, neighbors = brute_force.search(
            index, queries_device, k, **outputs
        )
        _check_results(distances, neighbors, expected, k, metric)
        if inplace:
            _check_results(
                outputs["distances"], outputs["neighbors"], expected, k, metric
            )


@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "cityblock", "inner_product"]
)
def test_brute_force_float64_precision(metric, tmp_path):
    # All dataset rows become identical when rounded to float32.
    dataset = np.zeros((6, 3), dtype=np.float64)
    dataset[:, 0] = 1.0 + np.array([5, 1, 3, 0, 4, 2]) * 1e-10
    queries = np.zeros((3, 3), dtype=np.float64)
    queries[:, 0] = [1.5, 2.0, 3.0]
    assert np.unique(dataset.astype(np.float32), axis=0).shape[0] == 1
    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)
    index = brute_force.build(dataset_device, metric)
    expected = _reference(dataset, queries, metric)
    distances, neighbors = brute_force.search(index, queries_device, 4)
    _check_results(distances, neighbors, expected, 4, metric)

    filename = str(tmp_path / "precise.bin")
    brute_force.save(filename, index)
    loaded = brute_force.load(filename)
    distances, neighbors = brute_force.search(loaded, queries_device, 4)
    _check_results(distances, neighbors, expected, 4, metric)


@pytest.mark.parametrize("filter_type", ["bitmap", "bitset"])
@pytest.mark.parametrize("density", [0.5, 0.05])
@pytest.mark.parametrize("n_queries", [3, 12])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "euclidean", "cosine", "inner_product"]
)
def test_brute_force_float64_filter(filter_type, density, n_queries, metric):
    rng = np.random.default_rng(123)
    n_samples, dim, k = 257, 17, 5
    dataset = rng.random((n_samples, dim))
    queries = rng.random((n_queries, dim))
    n_filter_rows = n_queries if filter_type == "bitmap" else 1
    allowed = np.zeros((n_filter_rows, n_samples), dtype=bool)
    for row in allowed:
        row[rng.choice(n_samples, int(n_samples * density), replace=False)] = (
            True
        )
    # DLPack filters pack rows consecutively, even across word boundaries.
    flat = allowed.ravel()
    padded = np.pad(flat, (0, (-flat.size) % 32))
    packed = np.packbits(padded, bitorder="little").view(np.uint32)
    bits_device = device_ndarray(packed)
    prefilter = getattr(filters, f"from_{filter_type}")(bits_device)
    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)
    index = brute_force.build(dataset_device, metric)
    distances, neighbors = brute_force.search(
        index, queries_device, k, prefilter=prefilter
    )
    expected = _reference(dataset, queries, metric)
    expected = np.where(
        allowed, expected, -np.inf if metric == "inner_product" else np.inf
    )
    _check_results(distances, neighbors, expected, k, metric)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_brute_force_output_dtype_roundtrip(dtype, tmp_path):
    assert brute_force.Index().dtype is None
    dataset = device_ndarray(np.arange(24, dtype=dtype).reshape(8, 3))
    queries = device_ndarray(np.array([[2, 3, 4], [8, 9, 10]], dtype=dtype))
    index = brute_force.build(dataset)
    assert index.dtype == np.dtype(dtype)
    before = brute_force.search(index, queries, 3)
    filename = str(tmp_path / "index.bin")
    brute_force.save(filename, index)
    loaded = brute_force.load(filename)
    assert loaded.dtype == np.dtype(dtype)
    after = brute_force.search(loaded, queries, 3)
    distance_dtype = np.float64 if dtype == np.float64 else np.float32
    assert before[0].dtype == after[0].dtype == np.dtype(distance_dtype)
    for original, restored in zip(before, after):
        np.testing.assert_array_equal(
            original.copy_to_host(), restored.copy_to_host()
        )


@pytest.mark.parametrize(
    "index_dtype,query_dtype",
    [
        (np.float64, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float16),
    ],
)
def test_brute_force_float64_reject_query_dtype(index_dtype, query_dtype):
    dataset = device_ndarray(np.ones((8, 3), dtype=index_dtype))
    index = brute_force.build(dataset)
    queries = device_ndarray(np.ones((2, 3), dtype=query_dtype))
    with pytest.raises(TypeError, match="must match index dtype"):
        brute_force.search(index, queries, 3)


@pytest.mark.parametrize(
    "index_dtype,distance_dtype",
    [
        (np.float64, np.float32),
        (np.float32, np.float64),
        (np.float16, np.float64),
    ],
)
def test_brute_force_float64_reject_distance_dtype(
    index_dtype, distance_dtype
):
    dataset = device_ndarray(np.ones((8, 3), dtype=index_dtype))
    index = brute_force.build(dataset)
    queries = device_ndarray(np.ones((2, 3), dtype=index_dtype))
    distances = device_ndarray.empty((2, 3), dtype=distance_dtype)
    with pytest.raises(TypeError, match="dtype .* not supported"):
        brute_force.search(index, queries, 3, distances=distances)
