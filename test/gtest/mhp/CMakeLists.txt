# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# tested with a variable number of ranks
add_executable(
  mhp-tests
  mhp-tests.cpp
#  ../common/all.cpp
#  ../common/copy.cpp
#  ../common/distributed_vector.cpp
#  ../common/drop.cpp
#  ../common/enumerate.cpp
#  ../common/fill.cpp
#  ../common/for_each.cpp
#  ../common/iota.cpp
#  ../common/reduce.cpp
  ../common/slide_view.cpp
#  ../common/subrange.cpp
#  ../common/take.cpp
#  ../common/transform_view.cpp
#  ../common/zip.cpp
#  ../common/zip_local.cpp
#  alignment.cpp
#  distributed_vector.cpp
#  reduce.cpp
#  stencil.cpp
#  segmented.cpp
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
  quick-test
  mhp-tests.cpp
  ../common/reduce.cpp
)
target_link_libraries(
  quick-test
  GTest::gtest_main
  cxxopts
  DR::mpi
)
target_compile_definitions(quick-test PRIVATE QUICK_TEST)

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-tests-1 mhp-tests 1)
endif()
add_mpi_test(mhp-tests-2 mhp-tests 2)
add_mpi_test(mhp-tests-3 mhp-tests 3)
add_mpi_test(mhp-tests-4 mhp-tests 4)

if (ENABLE_SYCL)
  add_executable(
    mhp-sycl-tests

    # pulls in the common test
    mhp-tests.cpp
    )
  target_compile_definitions(mhp-sycl-tests PRIVATE TEST_MHP_SYCL)
  target_compile_options(mhp-sycl-tests PRIVATE -fsycl)
  target_link_libraries(
    mhp-sycl-tests
    GTest::gtest_main
    cxxopts
    DR::mpi
    )

  if (NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1
    add_mpi_test(mhp-sycl-tests-1 mhp-sycl-tests 1)
  endif()
  add_mpi_test(mhp-sycl-tests-2 mhp-sycl-tests 2)
  add_mpi_test(mhp-sycl-tests-3 mhp-sycl-tests 3)
  add_mpi_test(mhp-sycl-tests-4 mhp-sycl-tests 4)
endif()
