# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# tested with a variable number of ranks
add_executable(
  mhp-tests

  mhp-tests.cpp

  algorithms.cpp
  alignment.cpp
  distributed_vector.cpp
  stencil.cpp
)

target_link_libraries(
  mhp-tests
  GTest::gtest_main
  cxxopts
  DR::mpi
)

if (ENABLE_SYCL)
  target_compile_options(mhp-tests PRIVATE -fsycl)
endif()
cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-tests-1 mhp-tests 1)
endif()
add_mpi_test(mhp-tests-2 mhp-tests 2)
add_mpi_test(mhp-tests-3 mhp-tests 3)
add_mpi_test(mhp-tests-4 mhp-tests 4)
