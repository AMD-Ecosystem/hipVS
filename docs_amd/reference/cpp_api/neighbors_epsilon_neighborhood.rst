..
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Epsilon Neighborhood
====================

Epsilon neighborhood finds all neighbors within a given radius (epsilon) for each point in a dataset. Unlike k-nearest neighbors which finds a fixed number of neighbors, epsilon neighborhood finds all points within a specified distance threshold, making it particularly useful for density-based algorithms and graph construction.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/epsilon_neighborhood.hpp>``

namespace *cuvs::neighbors::epsilon_neighborhood*

L2-Squared Distance Operations
------------------------------

.. doxygengroup:: epsilon_neighborhood_cpp_l2
    :project: cuvs
    :members:
    :content-only:
