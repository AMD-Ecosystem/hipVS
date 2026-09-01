..
    MIT License

    Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

NN-Descent
==========

NN-Descent is a graph construction algorithm that builds an approximate k-nearest neighbors graph, commonly used as a fast way to build the initial graph for graph-based ANN indexes such as CAGRA.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <cuvs/neighbors/nn_descent.hpp>``

namespace *cuvs::neighbors::nn_descent*

Index build parameters
----------------------

.. doxygengroup:: nn_descent_cpp_index_params
    :project: cuvs
    :members:
    :content-only:

Index
-----

.. doxygengroup:: nn_descent_cpp_index
    :project: cuvs
    :members:
    :content-only:

Index build
-----------

.. doxygengroup:: nn_descent_cpp_index_build
    :project: cuvs
    :members:
    :content-only:
