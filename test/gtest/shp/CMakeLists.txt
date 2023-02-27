# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(
  shp-tests
  shp-tests.cpp
  containers.cpp
  algorithms.cpp
  views.cpp
  copy.cpp
)

target_link_libraries(
  shp-tests
  GTest::gtest_main
  DR::shp
  fmt::fmt
  cxxopts
)
target_compile_options(shp-tests PRIVATE -fsycl)
target_link_options(shp-tests PRIVATE -fsycl)
if(ENABLE_CUDA)
  target_compile_options(shp-tests PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
  target_link_options(shp-tests PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
endif()

function(add_shp_test test_name name)
  if (DISABLED_TESTS)
    add_test(NAME ${test_name}-disabled COMMAND ./${name} ${ARGN} --gtest_also_run_disabled_tests)
  else()
    add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  endif()
endfunction()

add_shp_test(shp-3 shp-tests --devicesCount 3)
add_shp_test(shp shp-tests --gtest_also_run_disabled_tests)
