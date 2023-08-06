# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# cmake-format: off
add_executable(
  mhp-bench
  mhp-bench.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
  ../common/inclusive_scan.cpp
  ../common/stream.cpp
  rooted.cpp
  stencil_1d.cpp
  stencil_2d.cpp
  chunk.cpp
  mdspan.cpp
  mpi.cpp)
# cmake-format: on

if(NOT ENABLE_CUDA)
  # black_scholes.cpp does not compile in CUDA because it uses std::log
  target_sources(mhp-bench PRIVATE ../common/black_scholes.cpp)
endif()

target_compile_definitions(mhp-bench PRIVATE BENCH_MHP)
target_link_libraries(mhp-bench benchmark::benchmark cxxopts DR::mpi)

# mhp-quick-bench is for development. By reducing the number of source files, it
# builds much faster. Change the source files to match what you need to test. It
# is OK to commit changes to the source file list.

# cmake-format: off
add_executable(mhp-quick-bench mhp-bench.cpp
  ../common/distributed_vector.cpp
  )
# cmake-format: on
target_compile_definitions(mhp-quick-bench PRIVATE BENCH_MHP)
target_link_libraries(mhp-quick-bench benchmark::benchmark cxxopts DR::mpi)

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if(NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1 30000 is mininum because
  # of static column size for stencil2D disable DPL benchmarks because we get
  # intermittent fails with: ONEAPI_DEVICE_SELECTOR=opencl:cpu mpirun -n 1
  # ./mhp-bench --vector-size 30000 --rows 100 --columns 100 --check
  add_mpi_test(
    mhp-bench-1 mhp-bench 1 --vector-size 30000 --rows 100 --columns 100
    --check)
  if(ENABLE_SYCL)
    add_mpi_test(
      mhp-bench-1-sycl mhp-bench 1 --vector-size 30000 --rows 100 --columns 100
      --check --benchmark_filter=-.*DPL.* --sycl)
  endif()
endif()
