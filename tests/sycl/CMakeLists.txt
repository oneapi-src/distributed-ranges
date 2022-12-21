# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(
  sycl-tests
  sycl-tests.cpp
)

target_link_libraries(
  sycl-tests
  GTest::gtest_main
  DR::shp
)
target_compile_options(sycl-tests PRIVATE -fsycl)
target_link_options(sycl-tests PRIVATE -fsycl)
if(ENABLE_CUDA)
  target_compile_options(sycl-tests PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
  target_link_options(sycl-tests PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
endif()

add_test(NAME shp COMMAND ./sycl-tests)
