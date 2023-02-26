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
  add_test(NAME ${test_name} COMMAND ./${name} --gtest_filter=-ShpTests.InclusiveScan ${ARGN})
endfunction()

add_shp_test(shp-3 shp-tests --devicesCount 3)
add_shp_test(shp shp-tests)
