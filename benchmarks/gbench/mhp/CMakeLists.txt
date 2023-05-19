# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  mhp-bench
  mhp-bench.cpp
  ../common/distributed_vector.cpp
  rooted.cpp
)
target_link_libraries(
  mhp-bench
  benchmark::benchmark
  cxxopts
  DR::mpi
)

if (ENABLE_SYCL)
  target_compile_options(mhp-bench PRIVATE -fsycl)
endif()

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-bench-1 mhp-bench 1)
endif()
