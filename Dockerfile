# syntax=docker/dockerfile:1.5

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

ARG UBUNTU="24.04"
ARG ROCM=7.0
ARG BASE=rocm/dev-ubuntu-${UBUNTU}:${ROCM}-complete
FROM ${BASE}

ENV ROCM=${ROCM}

#Ensures that if any stage in a pipe fails, that the entire RUN command fails
SHELL ["/bin/bash", "-exo", "pipefail", "-c"]

RUN <<EOT
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        sudo \
        vim \
        python3 \
        software-properties-common \
        git-all \
        bash-completion \
        ninja-build \
        libblas-dev \
        liblapack-dev \
        wget \
        ca-certificates \
        clang-format \
        clangd \
        cmake-curses-gui \
        libsuitesparse-dev \
        ssh \
        rpm \
        libclang-dev \
        rsync \
        ccache
EOT

WORKDIR /third_party_builds

RUN <<EOT
wget -q https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz
tar xzf ucx-1.17.0.tar.gz
cd ucx-1.17.0
./contrib/configure-release --prefix=/usr --with-rocm=/opt/rocm
make -j$(nproc)
make install
EOT

RUN <<EOT
wget -q https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.6.tar.bz2
bzip2 -d openmpi-5.0.6.tar.bz2
tar -xvf openmpi-5.0.6.tar
cd openmpi-5.0.6
./configure --prefix=/usr --with-ucx=/usr --with-rocm=/opt/rocm
make -j $(nproc)
make install
EOT

RUN <<EOT
apt remove -y --purge --auto-remove cmake || echo "CMake not found"
wget -q https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.sh
bash ./cmake-4.0.1-linux-x86_64.sh --skip-license --prefix=/usr/local
EOT

RUN <<EOT
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba /usr/local/bin/
EOT

WORKDIR /home

RUN rm -rf /third_party_builds
