# Building hipVS from source

hipVS currently provides C++, C, Python and Rust APIs. The following instructions provide steps to build and test hipVS from source files provided in the [https://github.com/ROCm-DS/hipVS](https://github.com/ROCm-DS/hipVS) repository.

## Tested GPUs

| AMD Instinct GPU    | Architecture | Wavefront Size | LLVM target |
|---------------------|--------------|----------------|-------------|
| MI210               | CDNA2        | 64             | gfx90a      |
| MI250               | CDNA2        | 64             | gfx90a      |
| MI250X              | CDNA2        | 64             | gfx90a      |
| MI300A              | CDNA3        | 64             | gfx942      |
| MI300X              | CDNA3        | 64             | gfx942      |

## Dependencies

hipVS builds against the AMD ROCm software stack, that is, the ROCm runtime, HIP compiler tool-chain, and a GPU driver that matches your ROCm version.

Install ROCm 7.0.2 or later, or the minimum version supported by the GPUs listed above, and ensure the `rocminfo` and `hipcc` commands are in your `PATH`. For more information, see [ROCm Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

| Name                                                                  | Version / Notes                              |
| ----------------------------------------------------------            | -------------------------------------------- |
| [`cmake`](https://cmake.org/)                                         | ≥ 3.31.0                                     |
| [`ninja`](https://ninja-build.org/)                                   | ≥ 1.11.1                                     |
| [`hipsolver`](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/)| Version that comes bundled with ROCm ≥ 7.0.2 |
| [`hipblas`](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/)    | Version that comes bundled with ROCm ≥ 7.0.2 |
| [`hipblaslt`](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)| Version that comes bundled with ROCm ≥ 7.0.2 |
| [`hiprand`](https://rocm.docs.amd.com/projects/hipRAND/en/latest/)    | Version that comes bundled with ROCm ≥ 7.0.2 |
| [`hipsparse`](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/)| Version that comes bundled with ROCm ≥ 7.0.2 |
| [`libblas-dev`](https://www.netlib.org/lapack/)                       | Tested with 3.12.0                           |
| [`liblapack-dev`](https://www.netlib.org/lapack/)                     | Tested with 3.12.0                           |
| [`SuiteSparse`](https://github.com/DrTimothyAldenDavis/SuiteSparse)   | Tested with 7.6.1                            |
| [`libopenblas-dev`](https://github.com/OpenMathLib/OpenBLAS)          | Tested with 0.3.26                           |
| **Additional Required Dependencies**                                  |                                              |
| **\***[`hipMM`](https://github.com/ROCm-DS/hipMM)                     | TBD                                          |
| **\***[`hipCollections`](https://github.com/ROCm/hipCollections)      | TBD                                          |
| **\***[`hipRAFT`](https://github.com/ROCm-DS/hipRAFT)                 | TBD                                          |
| **\***[`hipCUB`](https://github.com/ROCm/hipCUB)                      | Version that comes bundled with ROCm ≥ 7.0.2 |
| **\***[`rocThrust`](https://github.com/ROCm/rocThrust)                | Version that comes bundled with ROCm ≥ 7.0.2 |
| **\*\***[`OpenMP`](https://www.openmp.org/)                           | Version that comes bundled with ROCm ≥ 7.0.2 |
| **Optional Dependencies**                                             |                                              |
| [`RCCL`](https://github.com/ROCm/rccl)                                | Version that comes bundled with ROCm ≥ 7.0.2 |
| [`UCX`](https://github.com/openucx/ucx)                               | ≥ 1.17.0                                     |
| [`Googletest`](https://github.com/google/googletest)                  | ≥ 1.13.0                                     |
| [`Googlebench`](https://github.com/google/benchmark)                  | ≥ 1.13.0                                     |
| [`Doxygen`](https://github.com/doxygen/doxygen)                       | >=1.8.20                                     |

> `*` - If not found locally the CMake build system will attempt to download a compatible version using [ROCmDS-cmake](https://github.com/ROCm-DS/ROCmDS-cmake).
>
> `**` - The `OpenMP` toolchain is automatically installed as part of the standard ROCm installation and is available under `/opt/rocm-{version}/llvm`.

## Docker

hipVS also provides a `Dockerfile` that encapsulates all the above dependencies for development. Below are the instructions to create a container using this Dockerfile.

```bash
cd <REPO_ROOT>
# If BuildKit is not available.
DOCKER_BUILDKIT=1 docker build -t <HIPVS_DEV_IMAGE> .
# If BuildKit is available: "docker buildx build  -t  <HIPVS_DEV_IMAGE> ."
docker run -d -it --cap-add=SYS_PTRACE \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add=video \
       --ipc=host \
       --name <CONTAINER_NAME> \
       --init \
       --network=host \
       --security-opt seccomp=unconfined \
       -v <REPO_ROOT>:<REPO_ROOT> <HIPVS_DEV_IMAGE>  tail -f /dev/null
docker exec -it <CONTAINER_NAME> bash
```
This container will have all necessary packages installed to build and run hipVS properly. The preceding command will create an Ubuntu 24.04 container that has ROCm installed. If you wish to use Ubuntu 22.04, replace the docker build command with the following:

```bash
docker build --build-arg UBUNTU=22.04 -t <HIPVS_DEV_IMAGE> .
```

## Environment variables

```bash
export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake # Set CMAKE_PREFIX_PATH to point to the ROCm installation site
```

## C and C++ library

The core functionality of hipVS is implemented in HIP/C++ and its functionality is exposed through `libcuvs.so`. There is also the C library `libcuvs_c.so` which is a wrapper around the C++ library. The C library is used to provide a C API to the functionality of the C++ library.

### Building and installing using build.sh

A utility script `<HIPVS_ROOT>/build.sh` is provided and is the entry-point to building various components.

`build.sh` uses [ROCmDS-cmake](https://github.com/ROCm-DS/ROCmDS-cmake), which will automatically download any dependencies that are not already installed.

The following will download the required dependencies, build and install the hipVS CMake package to the configured CMake install prefix.

```bash
cd <HIPVS_ROOT>
./build.sh libcuvs
```

The installation destination directory follows a similar structure:

```bash
<HIPVS_INSTALLATION_ROOT>
|-- include # Public C and C++ headers
|   |--*
`-- lib
    |-- cmake
    |   |-- cuvs
    |   |   |-- cuvs-c_api-c-targets-release.cmake
    |   |   |-- cuvs-c_api-c-targets.cmake
    |   |   |-- cuvs-config-version.cmake
    |   |   |-- cuvs-config.cmake
    |   |   |-- cuvs-dependencies.cmake
    |   |   |-- cuvs-targets-release.cmake
    |   |   `-- cuvs-targets.cmake
    |   `-- hnswlib
    |       |-- hnswlib-config-version.cmake
    |       |-- hnswlib-config.cmake
    |       `-- hnswlib-targets.cmake
    |-- libcuvs.so        # C++ Shared Library
    |-- libcuvs_c.so      # C   Shared Library
    `-- libcuvs_static.a  # C++ Static Library
```

### C and C++ tests

Compile the tests using the `tests` argument passed to `build.sh`.

```bash
./build.sh tests
```

The tests are broken into algorithm categories, so you'll find several binaries in `cpp/build/gtests` named `*_TEST`.

```bash
<HIPVS_ROOT>/cpp/build/gtests/
|-- BRUTEFORCE_C_TEST
|-- CAGRA_C_TEST
|-- CLUSTER_TEST
|-- DISTANCE_C_TEST
|-- DISTANCE_TEST
|-- HNSW_C_TEST
|-- INTEROP_TEST
|-- IVF_FLAT_C_TEST
|-- IVF_PQ_C_TEST
|-- NEIGHBORS_ANN_BRUTE_FORCE_TEST
|-- NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST
|-- NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST
|-- NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST
|-- NEIGHBORS_ANN_CAGRA_TEST_BUGS
|-- NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST
|-- NEIGHBORS_ANN_IVF_FLAT_TEST
|-- NEIGHBORS_ANN_IVF_PQ_TEST
|-- NEIGHBORS_ANN_NN_DESCENT_TEST
|-- NEIGHBORS_ANN_VAMANA_TEST
|-- NEIGHBORS_DYNAMIC_BATCHING_TEST
|-- NEIGHBORS_HNSW_TEST
|-- NEIGHBORS_TEST
|-- PREPROCESSING_TEST
|-- SPARSE_TEST
|-- STATS_TEST
|-- cuvs_c_neighbors_test
|-- cuvs_c_test
```

For example, to run the distance module tests:

```bash
./cpp/build/gtests/DISTANCE_TEST
```

It can take significant time to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the `--limit-tests` option in `build.sh`:

```bash
./build.sh tests -n --limit-tests="cuvs_c_test;STATS_TEST"
```

Running all the tests using ctest:

```bash
cd <HIPVS_ROOT>/cpp/build/
ctest --test-dir ./tests # If "--limit-tests" is specified, only a subset of tests are built. To run all the tests remove "--limit-tests" when building through `build.sh`
```

### ccache and sccache

[`ccache`](https://ccache.dev/) and [`sccache`](https://github.com/mozilla/sccache) can be used to better cache parts of the build when rebuilding frequently, such as when working on a new feature. You can also use `ccache` or `sccache` with `build.sh`:

```bash
./build.sh libcuvs --cache-tool=ccache
```

### GPU architecture selection

#### --allgpuarch

Builds hipVS for all supported GPU architectures, increasing portability but also build time. You can also use --allgpuarch with `build.sh`:

```bash
./build.sh clean
./build.sh libcuvs tests --allgpuarch
```

> **Note**
> Always execute a clean build or manually delete the build directory before running this argument if a previous build is present, to prevent potential build conflicts or errors.

#### Compile only for a specified GPU architecture

When specifying target architectures, provide them as a single string enclosed in double quotes. For multiple architectures, separate each with a semicolon (;).

```bash
./build.sh libcuvs tests --gpu-arch="gfx90a;gfx942"
```

OR

```bash
./build.sh libcuvs tests --gpu-arch="gfx942"
```

> **Note**
> Do not specify both `--gpu-arch` and `--allgpuarch` flags in the same build command. Avoid using multiple separate `--gpu-arch` flags by combining all target architectures into one `--gpu-arch` option.

### Using CMake directly

The `build.sh` script wraps common CMake configuration options. For finer control, invoke `cmake` directly as shown below.

The `CMAKE_INSTALL_PREFIX` option instructs CMake to install `hipVS` into a specific location.

```bash
cd <HIPVS_ROOT>/cpp
mkdir -p build && rm -rf build/*
cd build
# Configuration stage
cmake -S .. \
      -G Ninja \
      -B . \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_HIP_ARCHITECTURES=NATIVE \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DCMAKE_CXX_COMPILER=hipcc
# Build the cuvs target
ninja cuvs
# Install
ninja install
```

## Python Library

hipVS Python packages have a dependency on `CuPy`, which requires the `ROCM_HOME` environment variable to be set to the base folder of the ROCm installation, typically `/opt/rocm`.

### Build and install hipvs python packages

#### [Step 1] Setup Conda environment

Conda environment scripts are provided for installing the necessary dependencies to build the Python libraries from source. It is preferred to use [`micromamba`](https://github.com/mamba-org/mamba) as it's a fully statically-linked, self-contained, executable. `micromamba` can be installed by following the instructions on this page: [Micromamba Installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```bash
cd <HIPVS_ROOT>
micromamba  env create --name hipvs  --file conda/environments/all_rocm_arch-x86_64.yaml
# To initialize the current bash shell, run:
eval "$(micromamba shell hook --shell bash)"
micromamba activate hipvs
```
It is recommended to build the python wheels in a conda environment built from `all_rocm_arch-x86_64.yaml`. It is also possible to use `venv` but it is up to the user to install all the required packages in the environment.

#### [Step 2] Building and installing hipVS python package

#### Using build.sh

The Python libraries can be built and installed using the build.sh script:

```bash
# From within the hipvs environment:
cd <HIPVS_ROOT>
# Build and install hipvs-python and required dependencies.
./build.sh libcuvs python
# Test installation
python
>> import cuvs # Import should succeed
```

#### Building and installing the wheels manually

```bash
# Install the cuvs CMake package in the hipvs conda environment
cd <HIPVS_ROOT>
./build.sh libcuvs

# Build and install the libcuvs python wheel
cd <HIPVS_ROOT>/python/libcuvs
pip install -v --no-build-isolation --disable-pip-version-check .

# Build and install the cuvs python wheel
cd <HIPVS_ROOT>/python/cuvs/
pip install -v --no-build-isolation --disable-pip-version-check .
# Test installation
python
>> import cuvs
```

### Running the python tests

```bash
#set ROCM_HOME path
export ROCM_HOME=<path to rcom home> (e.g /opt/rocm)

# From within the hipvs environment
cd <HIPVS_ROOT>/python/cuvs/
py.test -v -s
```

## Rust Library

Building and running the Rust bindings tests assumes that Cargo (the Rust package manager and build tool) is installed and available in your environment.
If Cargo is not installed, see the [Rustup documentation](https://rustup.rs/) for installation instructions.

The Rust library can be built and installed using the `build.sh` script. As a prerequisite, the `hipvs-rust` library depends on the `hipvs` C++ library(`libcuvs.so`) and C library(`libcuvs_c.so`).
The `hipvs` C++ library must be built and installed before building the Rust library. If using the `build.sh` script, the following command can be used to build and install the hipVS C++ and Rust libraries:

```bash
# From within a hipvs conda environment:
cd <HIPVS_ROOT>
# The following command will first build and install the hipvs C++ library into the conda environment, then build and install the hipvs rust library and example.
# It will also run the rust tests after building the library.
./build.sh libcuvs rust
```

> **Note**
> The Rust option invoked through `build.sh` will not only build the `hipvs` and `hipvs-sys` crates, but also run the tests after.

### Running the Rust example after building through `build.sh`

```bash
# From within the hipvs environment
LD_LIBRARY_PATH=$CONDA_PREFIX/lib <HIPVS_ROOT>/rust/target/debug/examples/cagra
```

## Packaging

### Packaging with build.sh

The following command will generate a debian: `hipvs_<VERSION>_amd64.deb` in `<HIPVS_ROOT>/cpp/build`.

```bash
./build.sh libcuvs package
```

### Custom cpack generators

To generate other types of packages such as `tar` or `RPM` packages:

```bash
# Configure
mkdir -p <HIPVS_ROOT>/cpp/build
rm -rf <HIPVS_ROOT>/cpp/build/*
cd <HIPVS_ROOT>/cpp/build
cmake -S .. \
      -G Ninja \
      -B . \
      -DCMAKE_INSTALL_PREFIX=install \
      -DCMAKE_HIP_ARCHITECTURES=NATIVE \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAFT_COMPILE_LIBRARY=ON \
      -DBUILD_TESTS=OFF \
      -DCMAKE_CXX_COMPILER=hipcc

# Install to staging area
ninja install

# Invoke cpack to generate package
cpack -G RPM # To generate a RPM package. hipvs-<version>-Linux.rpm will be created at <HIPVS_ROOT>/cpp/build.
cpack -G TGZ # To generate a TGZ package. hipvs-<version>-Linux.tar.gz will be created at <HIPVS_ROOT>/cpp/build.
```

## Building Documentation

Prepare the environment to build documentation using the following commands:

```bash
cd <HIPVS_ROOT>
# Activate the hipvs conda environment.
micromamba activate hipvs
# Install dependencies and tools required for generating documentation.
pip install -r docs_amd/sphinx/requirements.txt
```

### Use `build.sh` to generate documentation

```bash
cd <HIPVS_ROOT>
./build.sh docs clean
```

Navigate to `<HIPVS_ROOT>/docs_amd/_build` and use the tool of your choice, e.g. Firefox, to open and examine the root level html file, `index.html`. From this point you should be able to navigate through the documentation.
