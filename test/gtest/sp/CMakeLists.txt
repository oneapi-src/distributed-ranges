# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  sp-tests
  sp-tests.cpp ../common/all.cpp ../common/copy.cpp ../common/counted.cpp
  ../common/distributed_vector.cpp ../common/drop.cpp ../common/enumerate.cpp
  ../common/equal.cpp ../common/fill.cpp ../common/for_each.cpp
  ../common/iota.cpp
  # ../common/iota_view.cpp
  ../common/reduce.cpp ../common/sort.cpp ../common/subrange.cpp
  ../common/take.cpp ../common/transform.cpp ../common/transform_view.cpp
  ../common/zip.cpp ../common/zip_local.cpp containers.cpp algorithms.cpp
  copy.cpp detail.cpp fill.cpp gemv.cpp transform.cpp)

add_executable(sp-tests-3 sp-tests.cpp containers-3.cpp copy-3.cpp)

# skeleton for rapid builds of individual tests, feel free to change this
add_executable(sp-quick-test sp-tests.cpp ../common/equal.cpp)
target_compile_definitions(sp-quick-test PRIVATE QUICK_TEST)

foreach(test-exec IN ITEMS sp-tests sp-tests-3 sp-quick-test)
  target_link_libraries(${test-exec} GTest::gtest_main DR::sp fmt::fmt cxxopts)
endforeach()

# timeout in SP tests
add_sp_ctest(sp-tests sp-tests)
add_sp_ctest(sp-tests-3 sp-tests --devicesCount 3)
add_sp_ctest(sp-tests-3-only sp-tests-3 --devicesCount 3)
add_sp_disabled_ctest(sp-quick-test sp-quick-test --devicesCount 3)
