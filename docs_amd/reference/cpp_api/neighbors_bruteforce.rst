Bruteforce
==========

The bruteforce method is running the KNN algorithm. It performs an extensive search, and in contrast to ANN methods produces an exact result.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/brute_force.hpp>``

Dense indices support ``half`` inputs with ``float`` distances, ``float`` inputs
with ``float`` distances, and ``double`` inputs with ``double`` distances.
Double indices retain FP64 precision in norms, distance arithmetic, selection,
and serialization. Dataset and query types must match; neighbor IDs are
``int64_t``. Sparse CSR indices also support ``double`` values and distances,
with ``int`` CSR indices and neighbor IDs.

Dense builds accept host row-major data or device row-major/column-major data.
Queries may use either device layout. Bitmap and bitset filtering requires
row-major queries and supports expanded L2 (squared or square root), inner
product, and cosine. These restrictions are the same for all supported types.

The dense ``index_params.metric_arg`` and sparse build metric argument remain
``float``; their values are promoted for double distance arithmetic. FP64 does
not require a particular AMD GPU architecture, although its throughput varies
by device. FP64 support here is limited to the C++ API; C and Python bindings
do not yet accept double inputs.

namespace *cuvs::neighbors::bruteforce*

Index
-----

.. doxygengroup:: bruteforce_cpp_index
    :project: cuvs
    :members:
    :content-only:

.. _neighbors-bruteforce-index-build:

Index build
-----------

.. doxygengroup:: bruteforce_cpp_index_build
    :project: cuvs
    :members:
    :content-only:

.. _neighbors-bruteforce-index-search:

Index search
------------

.. doxygengroup:: bruteforce_cpp_index_search
    :project: cuvs
    :members:
    :content-only:

Index serialize
---------------

.. doxygengroup:: bruteforce_cpp_index_serialize
    :project: cuvs
    :members:
    :content-only:
