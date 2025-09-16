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

ARG ROCM=7.0
FROM quay.io/pypa/manylinux_2_28_x86_64

ARG ROCM=7.0

RUN dnf install -y wget

RUN bash -c 'cat > /amdgpu_install_wrapper.sh' <<'EOF'
#!/usr/bin/env bash
set -exo pipefail

function get_os_property() {
  grep "^$1=" /etc/os-release | cut -d "=" -f 2 | tr -d "\"" | tr -d "'"
}

rhel_version_id=$(get_os_property VERSION_ID)
rhel_version_major=$(echo ${rhel_version_id} | cut -d '.' -f 1)

wget -np -r -nH --cut-dirs=4 -A "amdgpu-install*el${rhel_version_major}.noarch.rpm" https://repo.radeon.com/amdgpu-install/${ROCM}/rhel/${rhel_version_id}
yum clean all
yum install -y ./amdgpu-install*.rpm
rm ./amdgpu-install*.rpm

amdgpu-install -y --usecase="rocm,rocmdev,rocmdevtools,opencl,hip,openclsdk,hiplibsdk,openmpsdk,lrt,mllib,mlsdk" --no-dkms
# make ld aware about rocm install dir
echo "/opt/rocm/lib" > /etc/ld.so.conf.d/ROCM-MANUAL-INSTALL.conf
ldconfig
EOF

RUN export ROCM=${ROCM} && bash /amdgpu_install_wrapper.sh

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
wget -q https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz
tar xzf ucx-1.17.0.tar.gz
cd ucx-1.17.0
./contrib/configure-release --prefix=/usr --with-rocm=/opt/rocm
make -j$(nproc)
make install
EOT

RUN <<EOT
wget -q https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.bz2
bzip2 -d openmpi-5.0.7.tar.bz2
tar -xvf openmpi-5.0.7.tar
cd openmpi-5.0.7
export CFLAGS="-Wno-int-conversion -Wno-implicit-function-declaration -Wno-builtin-declaration-mismatch"
./configure --prefix=/usr --with-ucx=/usr --with-rocm=/opt/rocm
make -j $(nproc)
make install
EOT

RUN <<EOT
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba /usr/local/bin/
EOT

RUN rm -rf /third_party_builds
