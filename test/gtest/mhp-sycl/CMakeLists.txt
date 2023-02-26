# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# tested with a variable number of ranks
add_executable(
  mhp-sycl-tests-n

  distributed_vector.cpp
  mhp-sycl-tests.cpp
)

target_compile_options(mhp-sycl-tests-n PRIVATE -fsycl)

target_link_libraries(
  mhp-sycl-tests-n
  GTest::gtest_main
  cxxopts
  DR::mpi
)

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-sycl-tests-1 mhp-sycl-tests-n 1)
endif()
add_mpi_test(mhp-sycl-tests-2 mhp-sycl-tests-n 2)
add_mpi_test(mhp-sycl-tests-3 mhp-sycl-tests-n 3)
add_mpi_test(mhp-sycl-tests-4 mhp-sycl-tests-n 4)
