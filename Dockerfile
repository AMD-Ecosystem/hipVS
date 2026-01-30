# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# syntax=docker/dockerfile:1.5

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# <<<<<<< OLD CODE from 09b9a06bf (6f80caaa1) - COMMENTED OUT
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
#
# ARG UBUNTU="24.04"
# ARG ROCM=7.2.4
# ARG BASE=rocm/dev-ubuntu-${UBUNTU}:${ROCM}-complete
# FROM ${BASE}
#
# ENV ROCM=${ROCM}
#
# #Ensures that if any stage in a pipe fails, that the entire RUN command fails
# SHELL ["/bin/bash", "-exo", "pipefail", "-c"]
#
# RUN <<EOT
# apt-get update
# DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#         sudo \
#         vim \
#         python3 \
#         software-properties-common \
#         git-all \
#         bash-completion \
#         ninja-build \
#         libblas-dev \
#         libopenblas-dev \
#         liblapack-dev \
#         wget \
#         ca-certificates \
#         clang-format \
#         clangd \
#         cmake-curses-gui \
#         libsuitesparse-dev \
#         ssh \
#         rpm \
#         libclang-dev \
#         rsync \
#         ccache
# EOT
#
# WORKDIR /third_party_builds
#
# RUN <<EOT
# apt remove -y --purge --auto-remove cmake || echo "CMake not found"
# wget -q https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.sh
# bash ./cmake-4.0.1-linux-x86_64.sh --skip-license --prefix=/usr/local
# EOT
#
# RUN <<EOT
# curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
# mv bin/micromamba /usr/local/bin/
# EOT
#
# WORKDIR /home
#
# RUN rm -rf /third_party_builds
# =======
# Customizable Build Arguments:
#   docker build --build-arg CUDA_VER=12.8 --build-arg RAPIDS_VER=25.06 -t cuvs:custom .
# ==============================================================================

# Configurable build arguments - override with --build-arg during build
ARG CUDA_VER=12.9.1

# Python version for the conda environment (supported: 3.11, 3.12, 3.13)
ARG PYTHON_VER=3.12

# RAPIDS/cuVS version
ARG RAPIDS_VER=25.06

FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04

# Display build configuration for verification
RUN echo "  Building cuVS Docker image with:" && \
    echo "   CUDA Version: ${CUDA_VER}" && \
    echo "   Python Version: ${PYTHON_VER}" && \
    echo "   RAPIDS Version: ${RAPIDS_VER}"

# Container metadata
LABEL maintainer="RAPIDS cuVS Team"
LABEL description="RAPIDS cuVS - Vector Search and Clustering on GPU"
LABEL org.opencontainers.image.source="https://github.com/rapidsai/cuvs"
LABEL org.opencontainers.image.usage="docker run --gpus all -it <image>"

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (lightweight conda with conda-forge defaults)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

# Create conda environment and install cuVS packages with pinned dependencies
RUN conda create -n cuvs python=${PYTHON_VER} -y && \
    conda run -n cuvs conda install -c rapidsai -c conda-forge \
        cuvs=${RAPIDS_VER} \
        libcuvs=${RAPIDS_VER} \
        cuda-version=${CUDA_VER} \
        "numpy>=1.23,<3.0a0" \
        "cupy>=12.0.0" \
        jupyter \
        ipython \
        -y && \
    conda clean -afy

# Create non-root user
RUN useradd -m -s /bin/bash cuvs
USER cuvs
WORKDIR /home/cuvs

# Initialize conda for the user and configure environment activation
RUN ${CONDA_DIR}/bin/conda init bash && \
    echo "conda activate cuvs" >> ~/.bashrc && \
    echo "conda activate cuvs" >> ~/.bash_profile

# Set environment variables for automatic cuvs environment activation
ENV PATH=${CONDA_DIR}/envs/cuvs/bin:$PATH
ENV CONDA_DEFAULT_ENV=cuvs
ENV CONDA_PREFIX=${CONDA_DIR}/envs/cuvs

# Verify installation and show version info
RUN python -c "import cuvs; print('cuVS version:', cuvs.__version__)"

# Default command
CMD ["bash"]
# >>>>>>> END NEW CODE (6f80caaa1)
