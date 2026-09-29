# syntax=docker/dockerfile:1

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

FROM quay.io/pypa/manylinux_2_28_x86_64

ARG ROCM=10.0.0

RUN dnf install -y wget

# ROCm 10 is no longer shipped through amdgpu-install; it comes from the
# amdrocm dnf repo and installs into /opt/rocm/core-<major.minor>, with
# update-alternatives symlinks for /opt/rocm/{bin,lib,include,llvm,...}.
RUN bash -c 'cat > /rocm_install_wrapper.sh' <<'EOF'
#!/usr/bin/env bash
set -exo pipefail

rocm_series=${ROCM%.*}

tee /etc/yum.repos.d/amdrocm-stable.repo <<REPO
[amdrocm-stable]
name=ROCm ${ROCM}
baseurl=https://stable.repo.amd.com/rocm/core/packages/rhel8/x86_64
enabled=1
gpgcheck=1
gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
REPO
dnf clean all

dnf install -y libatomic libquadmath
dnf install -y "amdrocm-core-sdk${rocm_series}-${ROCM}"
# make ld aware about rocm install dir
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/ROCM-MANUAL-INSTALL.conf
ldconfig
EOF

RUN export ROCM=${ROCM} && bash /rocm_install_wrapper.sh

RUN dnf install -y sudo \
        vim \
        python3 \
        git-all \
        bash-completion \
        ninja-build \
        ca-certificates \
        clangd \
        ccache \
        openblas-devel \
        lapack-devel \
        suitesparse-devel

WORKDIR /third_party_builds

RUN <<EOT
wget -q https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.sh
bash ./cmake-4.0.1-linux-x86_64.sh --skip-license --prefix=/usr/local
EOT


RUN <<EOT
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba /usr/local/bin/
EOT

RUN rm -rf /third_party_builds
