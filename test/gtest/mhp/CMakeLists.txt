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
  ../common/counted.cpp
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
  mdstar.cpp
  reduce.cpp
  stencil.cpp
  segments.cpp
  slide_view.cpp
)

add_executable(
  mhp-tests-3
  mhp-tests.cpp
  slide_view-3.cpp
  distributed_dense_matrix3.cpp
)

# mhp-quick-test is for development. By reducing the number of source
# files, it builds much faster. Change the source files to match what
# you need to test. It is OK to commit changes to the source file
# list.

add_executable(
  mhp-quick-test
  mhp-tests.cpp
  mdstar.cpp
)

target_compile_definitions(mhp-quick-test PRIVATE QUICK_TEST)

foreach(test-exec IN ITEMS mhp-tests mhp-tests-3 mhp-quick-test)
  target_link_libraries(
    ${test-exec}
    GTest::gtest_main
    cxxopts
    DR::mpi
  )
endforeach()

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(mhp-tests-1 mhp-tests 1)
endif()
add_mpi_test(mhp-tests-2 mhp-tests 2)
add_mpi_test(mhp-tests-3 mhp-tests 3)
add_mpi_test(mhp-tests-4 mhp-tests 4)
add_mpi_test(mhp-tests-3-only mhp-tests-3 3)

if (ENABLE_SYCL)
  if (NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1
    add_mpi_test(mhp-sycl-tests-1 mhp-tests 1 --sycl --gtest_filter=-*Slide*)
  endif()
  #TODO: fix sycl Slide issues, see https://github.com/oneapi-src/distributed-ranges/issues/322
  add_mpi_test(mhp-sycl-tests-2 mhp-tests 2 --sycl --gtest_filter=-*Slide*)
  add_mpi_test(mhp-sycl-tests-3 mhp-tests 3 --sycl --gtest_filter=-*Slide*)
  add_mpi_test(mhp-sycl-tests-4 mhp-tests 4 --sycl --gtest_filter=-*Slide*)
  add_mpi_test(mhp-sycl-tests-3-only mhp-tests-3 3 --sycl --gtest_filter=-*Slide*)
endif()
