# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# tested with a variable number of ranks
add_executable(
  mhp-tests-n

  mhp-tests.cpp

  algorithms.cpp
  alignment.cpp
  distributed_vector.cpp
  stencil.cpp
  views.cpp
)

target_link_libraries(
  mhp-tests-n
  GTest::gtest_main
  cxxopts
  DR::mpi
)

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-mpi-1 mhp-tests-n 1)
endif()
add_mpi_test(mhp-mpi-2 mhp-tests-n 2)
add_mpi_test(mhp-mpi-3 mhp-tests-n 3)
add_mpi_test(mhp-mpi-4 mhp-tests-n 4)
