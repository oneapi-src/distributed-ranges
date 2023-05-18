# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# tested with a variable number of ranks
add_executable(
  mhp-tests
  mhp-tests.cpp
  ../common/all.cpp
  ../common/copy.cpp
  ../common/distributed_vector.cpp
  ../common/drop.cpp
  ../common/enumerate.cpp
  ../common/fill.cpp
  ../common/for_each.cpp
  ../common/iota.cpp
  ../common/reduce.cpp
  ../common/subrange.cpp
  ../common/take.cpp
  ../common/transform.cpp
  ../common/transform_view.cpp
  ../common/zip.cpp
  ../common/zip_local.cpp
  alignment.cpp
  copy.cpp
  distributed_vector.cpp
  distributed_dense_matrix.cpp
  reduce.cpp
  stencil.cpp
  segmented.cpp
)
target_link_libraries(
  mhp-tests
  GTest::gtest_main
  cxxopts
  DR::mpi
)

# skeleton for rapid builds of individual tests, feel free to change
# this
add_executable(
  mhp-quick-test
  mhp-tests.cpp
  ../common/take.cpp
)
target_link_libraries(
  mhp-quick-test
  GTest::gtest_main
  cxxopts
  DR::mpi
)
target_compile_definitions(mhp-quick-test PRIVATE QUICK_TEST)

if (ENABLE_SYCL)
  target_compile_options(mhp-tests PRIVATE -fsycl)
  target_compile_options(mhp-quick-test PRIVATE -fsycl)
endif()

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-tests-1 mhp-tests 1)
endif()
add_mpi_test(mhp-tests-2 mhp-tests 2)
add_mpi_test(mhp-tests-3 mhp-tests 3)
add_mpi_test(mhp-tests-4 mhp-tests 4)

if (ENABLE_SYCL)
  if (NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1
    add_mpi_test(mhp-sycl-tests-1 mhp-tests 1 --sycl --gtest_filter=-Take*)
  endif()
  # TODO fix sycl Take issues
  add_mpi_test(mhp-sycl-tests-2 mhp-tests 2 --sycl)
  add_mpi_test(mhp-sycl-tests-3 mhp-tests 3 --sycl)
  add_mpi_test(mhp-sycl-tests-4 mhp-tests 4 --sycl)
endif()
