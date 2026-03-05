<!---
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
-->

<head>
  <meta charset="UTF-8">
  <meta name="description" content="hipVS documentation and API reference library">
  <meta name="keywords" content="Machine-Learning, Vector Search, Primitives,GPU, RAPIDS, ROCm-DS">
</head>

# Installing hipVS

You can install hipVS via AMD PyPI as described below. This is recommended for users of hipVS. For developers interested in modifying or contributing to the Open Source hipVS component, see the [Build instructions](./build.md).

## Requirements

hipVS requires ROCm 7.2.1 running on a [ROCm-supported operating system](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/reference/system-requirements.html#supported-operating-systems). Using Ubuntu 22.04 or later is recommended.
For more information, see [ROCm-DS system requirements](https://rocm.docs.amd.com/projects/rocm-ds-internal/en/latest/install/install.html).

The steps in this topic require a Conda installation. A minimal free version of Conda is [Miniforge](https://conda-forge.org/download/).

## Install hipVS via AMD PyPI

Packaged versions of hipVS and its dependencies are distributed via
[AMD PyPI](https://pypi.amd.com/rocm-7.2.1/simple/). This section discusses how to install
hipVS via this package index.

Create and activate a Conda environment with Python 3.12 as shown below:

```bash
conda create --name hipVS python=3.12
conda activate hipVS
```

hipVS can then be installed into this environment using pip and the AMD PyPI URL:

```bash
pip install amd-hipvs==1.0.0 --extra-index-url=https://pypi.amd.com/rocm-7.2.1/simple/
```
