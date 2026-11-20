# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

FROM ubuntu:26.04

# Find a nightly tarball from https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-release-tarballs
# Full list of tarballs: https://therock-nightly-tarball.s3.amazonaws.com/index.html
# Note: The tarballs are architecture specific.
# Example: https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx90X-dcgpu-7.11.0a20260109.tar.gz
ARG THE_ROCK_NIGHTLY_TARBALL_URL

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
        libopenblas-dev \
        wget \
        ca-certificates \
        clang-format \
        clangd \
        cmake-curses-gui \
        libsuitesparse-dev \
        ssh \
        rpm \
        ccache \
        build-essential \
        curl \
        autoconf \
        libtool \
        rdma-core \
        rdmacm-utils \
        librdmacm-dev \
        libibverbs1 \
        ibutils \
        infiniband-diags
EOT

WORKDIR /third_party_builds

RUN <<EOT
wget -q ${THE_ROCK_NIGHTLY_TARBALL_URL} -O therock-nightly.tar.gz
mkdir -p /opt/rocm
tar -xzf therock-nightly.tar.gz -C /opt/rocm --strip-components=1

tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm/lib
/opt/rocm/lib64
EOF
ldconfig

tee /etc/profile.d/set-rocm-env.sh << EOF
export PATH=\$PATH:/opt/rocm/bin
EOF

rm therock-nightly.tar.gz
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
