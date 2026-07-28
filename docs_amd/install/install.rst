..
    MIT License

    Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

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

.. meta::
   :description: hipVS documentation and API reference library
   :keywords: Machine-Learning, Vector Search, Primitives,GPU, RAPIDS, AMD Data Science

****************
Installing hipVS
****************

You can install hipVS via AMD PyPI as described below. This is recommended for users of hipVS. For developers interested in modifying or contributing to the Open Source hipVS component, see the :doc:`Build instructions <./build>`.

See :ref:`sys-req` for information regarding supported operating systems, ROCm versions,
and AMD GPUs before installing hipVS.

Install hipVS via AMD PyPI
==========================

Packaged versions of hipVS and its dependencies are distributed via
`AMD PyPI <https://pypi.amd.com/rocm-7.2.3/simple/>`__. This section discusses how to install
hipVS via this package index.

Create and activate a Conda environment with a compatible Python version, such as 3.11 or 3.12 as shown below. For more information on compatible Python versions, see :ref:`sys-req`.

.. code-block:: bash

   conda create --name hipVS python=3.12 # Specify your Python version
   conda activate hipVS

hipVS can then be installed into this environment using pip and the AMD PyPI URL:

.. code-block:: bash

   pip install amd-hipvs==1.0.0 --extra-index-url=https://pypi.amd.com/rocm-7.2.3/simple/
