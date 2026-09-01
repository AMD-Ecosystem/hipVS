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

Preprocessing
=============

.. role:: py(code)
   :language: python
   :class: highlight

Binary Quantizer
################

.. autoapifunction:: cuvs.preprocessing.quantize.binary.transform

Product Quantizer
#################

.. autoapiclass:: cuvs.preprocessing.quantize.pq.Quantizer
    :members:

.. autoapiclass:: cuvs.preprocessing.quantize.pq.QuantizerParams
    :members:

.. autoapifunction:: cuvs.preprocessing.quantize.pq.build

.. autoapifunction:: cuvs.preprocessing.quantize.pq.transform

.. autoapifunction:: cuvs.preprocessing.quantize.pq.inverse_transform

Scalar Quantizer
################

.. autoapiclass:: cuvs.preprocessing.quantize.scalar.Quantizer
    :members:

.. autoapiclass:: cuvs.preprocessing.quantize.scalar.QuantizerParams
    :members:

.. autoapifunction:: cuvs.preprocessing.quantize.scalar.train

.. autoapifunction:: cuvs.preprocessing.quantize.scalar.transform

.. autoapifunction:: cuvs.preprocessing.quantize.scalar.inverse_transform

Principal Component Analysis
#############################

.. autoapiclass:: cuvs.preprocessing.pca.Params
    :members:

.. autoapifunction:: cuvs.preprocessing.pca.fit

.. autoapifunction:: cuvs.preprocessing.pca.fit_transform

.. autoapifunction:: cuvs.preprocessing.pca.transform

.. autoapifunction:: cuvs.preprocessing.pca.inverse_transform
