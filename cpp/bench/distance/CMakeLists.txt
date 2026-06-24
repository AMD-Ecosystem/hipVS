# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if(BUILD_CPU_ONLY)
  return()
endif()

add_executable(pairwise_distance_bench pairwise_distance_bench.cu)

target_link_libraries(
  pairwise_distance_bench PRIVATE cuvs cuvs::cuvs raft::raft benchmark::benchmark Threads::Threads
                                  stdc++fs
)
# Hide std::filesystem symbols statically linked from libstdc++fs.a so they do not interpose the
# implementations inside ROCm shared libraries compiled against a newer GCC ABI.
target_link_options(pairwise_distance_bench PRIVATE "-Wl,--exclude-libs,libstdc++fs.a")

set_target_properties(
  pairwise_distance_bench
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CUVS_BINARY_DIR}/bench/distance"
             BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
)

target_compile_options(
  pairwise_distance_bench
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
          "$<$<OR:$<COMPILE_LANGUAGE:CUDA>,$<COMPILE_LANGUAGE:HIP>>:${CUVS_GPU_FLAGS}>"
)

target_include_directories(
  pairwise_distance_bench PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
)

install(
  TARGETS pairwise_distance_bench
  COMPONENT bench
  DESTINATION bin/bench/distance
)

add_executable(fused_distance_nn_bench fused_distance_nn_bench.cu)

target_link_libraries(
  fused_distance_nn_bench PRIVATE cuvs cuvs::cuvs raft::raft cuvs_src_internal benchmark::benchmark
                                  Threads::Threads stdc++fs
)
target_link_options(fused_distance_nn_bench PRIVATE "-Wl,--exclude-libs,libstdc++fs.a")

set_target_properties(
  fused_distance_nn_bench
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "${CUVS_BINARY_DIR}/bench/distance"
             BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
)

target_compile_options(
  fused_distance_nn_bench
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
          "$<$<OR:$<COMPILE_LANGUAGE:CUDA>,$<COMPILE_LANGUAGE:HIP>>:${CUVS_GPU_FLAGS}>"
)

target_include_directories(
  fused_distance_nn_bench PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
)

install(
  TARGETS fused_distance_nn_bench
  COMPONENT bench
  DESTINATION bin/bench/distance
)
