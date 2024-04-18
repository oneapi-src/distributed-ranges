# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# tested with a variable number of ranks
# cmake-format: off
add_executable(
  mhp-tests
  mhp-tests.cpp
  ../common/all.cpp
  ../common/copy.cpp
  ../common/counted.cpp
  ../common/distributed_vector.cpp
  ../common/drop.cpp
  ../common/enumerate.cpp
  ../common/exclusive_scan.cpp
  ../common/fill.cpp
  ../common/for_each.cpp
  ../common/inclusive_scan.cpp
  ../common/iota.cpp
  ../common/iota_view.cpp
  ../common/reduce.cpp
  ../common/sort.cpp
  ../common/subrange.cpp
  ../common/sycl_utils.cpp
  ../common/take.cpp
  ../common/transform.cpp
  ../common/transform_view.cpp
  ../common/zip.cpp
  ../common/zip_local.cpp
  alignment.cpp
  communicator.cpp
  copy.cpp
  distributed_vector.cpp
  halo.cpp
  mdstar.cpp
  mhpsort.cpp
  reduce.cpp
  stencil.cpp
  segments.cpp
  slide_view.cpp
  wave_kernel.cpp)

add_executable(
  mhp-tests-3
  mhp-tests.cpp
  communicator-3.cpp
  halo-3.cpp
  slide_view-3.cpp)

# mhp-quick-test is for development. By reducing the number of source files, it
# builds much faster. Change the source files to match what you need to test. It
# is OK to commit changes to the source file list.

add_executable(mhp-quick-test
  mhp-tests.cpp
  ../common/sort.cpp
  )
# cmake-format: on

target_compile_definitions(mhp-quick-test PRIVATE QUICK_TEST)

foreach(test-exec IN ITEMS mhp-tests mhp-tests-3 mhp-quick-test)
  if(ENABLE_ISHMEM)
    target_link_ishmem(${test-exec})
  endif()
  target_link_libraries(${test-exec} GTest::gtest_main cxxopts DR::mpi)
  set_property(TARGET ${test-exec} PROPERTY RULE_LAUNCH_COMPILE
                                            "${CMAKE_COMMAND} -E time")
endforeach()

# tests without --sycl flag will fail on IshmemBackend TODO: make them be
# running somehow if ENABLE_ISHMEM will be default CI config
if(NOT ENABLE_ISHMEM)
  add_mhp_ctest(NAME mhp-quick-test NPROC 1)
  add_mhp_ctest(NAME mhp-quick-test NPROC 2)

  cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

  if(NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1
    add_mhp_ctest(NAME mhp-tests NPROC 1)
  endif()
  foreach(nproc RANGE 2 4)
    add_mhp_ctest(NAME mhp-tests NPROC ${nproc})
  endforeach()
  add_mhp_ctest(TEST_NAME mhp-tests-3-only NAME mhp-tests-3 NPROC 3)
endif()

if(ENABLE_SYCL)
  # DRA-83: Slide isn't complete
  set(sycl-exclusions *Slide*:ComplexSlide*:)

  if(ENABLE_ISHMEM)
    # DRA-84 some Halo3 cases don't work on IshmemBackend (work on MPI)
    set(sycl-exclusions ${sycl-exclusions}Halo3/*:)
  endif()

  add_mhp_ctest(NAME mhp-quick-test NPROC 1 SYCL)
  add_mhp_ctest(NAME mhp-quick-test NPROC 2 SYCL)
  add_mhp_ctest(
    NAME mhp-quick-test NPROC 1 OFFLOAD SYCL TARGS --device-memory)
  add_mhp_ctest(
    NAME mhp-quick-test NPROC 2 OFFLOAD SYCL TARGS --device-memory)

  add_mhp_ctest(
    NAME mhp-tests NPROC 2 OFFLOAD SYCL TARGS --device-memory
    --gtest_filter=-${sycl-exclusions})
  if(NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1
    add_mhp_ctest(
      NAME mhp-tests NPROC 1 SYCL TARGS --gtest_filter=-${sycl-exclusions})
  endif()
  # TODO: fix sycl Slide issues, see
  # https://github.com/oneapi-src/distributed-ranges/issues/322

  foreach(nproc RANGE 2 4)
    add_mhp_ctest(
      NAME mhp-tests NPROC ${nproc} SYCL TARGS
      --gtest_filter=-${sycl-exclusions})
  endforeach()
  add_mhp_ctest(
    TEST_NAME mhp-tests-sycl-3-only NAME mhp-tests-3 NPROC 3 SYCL TARGS
    --gtest_filter=-${sycl-exclusions})
endif()
