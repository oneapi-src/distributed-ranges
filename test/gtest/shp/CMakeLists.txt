# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  shp-tests
  shp-tests.cpp ../common/all.cpp ../common/copy.cpp ../common/counted.cpp
  ../common/distributed_vector.cpp ../common/drop.cpp ../common/enumerate.cpp
  ../common/fill.cpp ../common/for_each.cpp ../common/iota.cpp
  # ../common/iota_view.cpp
  ../common/reduce.cpp ../common/sort.cpp ../common/subrange.cpp
  ../common/take.cpp ../common/transform.cpp ../common/transform_view.cpp
  ../common/zip.cpp ../common/zip_local.cpp containers.cpp algorithms.cpp
  copy.cpp fill.cpp gemv.cpp transform.cpp)

add_executable(shp-tests-3 shp-tests.cpp containers-3.cpp copy-3.cpp)

# skeleton for rapid builds of individual tests, feel free to change this
add_executable(shp-quick-test shp-tests.cpp ../common/transform.cpp)
target_compile_definitions(shp-quick-test PRIVATE QUICK_TEST)

foreach(test-exec IN ITEMS shp-tests shp-tests-3 shp-quick-test)
  target_link_libraries(${test-exec} GTest::gtest_main DR::shp fmt::fmt cxxopts)
endforeach()

function(add_shp_test test_name name)
  if(DISABLED_TESTS)
    add_test(NAME ${test_name}-disabled COMMAND ./${name} ${ARGN}
                                                --gtest_also_run_disabled_tests)
  else()
    add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  endif()
endfunction()

add_shp_test(shp shp-tests)
add_shp_test(shp-3 shp-tests --devicesCount 3)
add_shp_test(shp-3-only shp-tests-3 --devicesCount 3)
