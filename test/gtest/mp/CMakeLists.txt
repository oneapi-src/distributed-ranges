# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# tested with a variable number of ranks
# cmake-format: off
add_executable(
  mp-tests
  mp-tests.cpp
  ../common/all.cpp
  ../common/copy.cpp
  ../common/count.cpp
  ../common/counted.cpp
  ../common/distributed_vector.cpp
  ../common/drop.cpp
  ../common/enumerate.cpp
  ../common/equal.cpp
#  ../common/exclusive_scan.cpp disabled due to deadlock - DRA-213
  ../common/fill.cpp
  ../common/for_each.cpp
#  ../common/inclusive_scan.cpp disabled due to deadlock - DRA-213
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
  dual_distributed_vector.cpp
  halo.cpp
  mdstar.cpp
  mpsort.cpp
  reduce.cpp
  stencil.cpp
  segments.cpp
  slide_view.cpp
  wave_kernel.cpp)

add_executable(
  mp-tests-3
  mp-tests.cpp
  communicator-3.cpp
  halo-3.cpp
  slide_view-3.cpp)

# mp-quick-test is for development. By reducing the number of source files, it
# builds much faster. Change the source files to match what you need to test. It
# is OK to commit changes to the source file list.

add_executable(mp-quick-test
  mp-tests.cpp
  halo.cpp
  # ../common/for_each.cpp
  # dual_distributed_vector.cpp
  )
# cmake-format: on

target_compile_definitions(mp-quick-test PRIVATE QUICK_TEST DR_FORMAT)

foreach(test-exec IN ITEMS mp-tests mp-tests-3 mp-quick-test)
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
  add_mp_ctest(NAME mp-quick-test NPROC 1)
  add_mp_ctest(NAME mp-quick-test NPROC 2)

  cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

  if(NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1
    add_mp_ctest(NAME mp-tests NPROC 1 TIMEOUT 150)
  endif()
  foreach(nproc RANGE 2 4)
    add_mp_ctest(NAME mp-tests NPROC ${nproc} TIMEOUT 150)
  endforeach()
  add_mp_ctest(
    TEST_NAME mp-tests-3-only NAME mp-tests-3 NPROC 3 TIMEOUT 150)
endif()

if(ENABLE_SYCL)
  # DRA-83: Slide isn't complete
  set(sycl-exclusions *Slide*:ComplexSlide*:)

  if(ENABLE_ISHMEM)
    # Some Halo3 cases don't work on IshmemBackend (work on MPI, see: DRA-84),
    # Counted.large fails with
    # distributed-ranges/test/gtest/common/counted.cpp:62: Failure Expected
    # equality of these values: *(--counted_result.end()) Which is: 5, should be
    # 77 Mdspan, Mdarray hangs sometimes on ISHMEM.
    set(sycl-exclusions
        ${sycl-exclusions}Halo3/*:Sort*:Counted/*:Mdspan*:Mdarray*:)
  endif()

  foreach(nproc RANGE 1 4)
    add_mp_ctest(NAME mp-quick-test NPROC ${nproc} SYCL)
    add_mp_pvc_ctest(
      NAME mp-quick-test NPROC ${nproc} OFFLOAD SYCL TARGS --device-memory)
  endforeach()

  add_mp_pvc_ctest(
    NAME mp-tests NPROC 2 TIMEOUT 150 OFFLOAD SYCL TARGS --device-memory
    --gtest_filter=-${sycl-exclusions})
  if(NOT MPI_IMPL STREQUAL "openmpi")
    # MPI_Win_create fails for communicator with size 1 this should be
    # add_mp_ctest - DRA-215
    add_mp_pvc_ctest(
      NAME mp-tests NPROC 1 SYCL TARGS --gtest_filter=-${sycl-exclusions})
  endif()
  # TODO: fix sycl Slide issues, see
  # https://github.com/oneapi-src/distributed-ranges/issues/322

  foreach(nproc RANGE 2 4)
    add_mp_ctest(
      NAME mp-tests NPROC ${nproc} TIMEOUT 500 SYCL TARGS
      --gtest_filter=-${sycl-exclusions})
  endforeach()
  add_mp_ctest(
    TEST_NAME mp-tests-sycl-3-only NAME mp-tests-3 NPROC 3 TIMEOUT 500 SYCL
    TARGS --gtest_filter=-${sycl-exclusions})
endif()
