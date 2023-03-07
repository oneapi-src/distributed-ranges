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

add_executable(
  shp-tests-3
  shp-tests.cpp
  containers-3.cpp
  copy-3.cpp
)

foreach(test-exec IN ITEMS shp-tests shp-tests-3)
  target_link_libraries(
    ${test-exec}
    GTest::gtest_main
    DR::shp
    fmt::fmt
    cxxopts
  )

  target_compile_options(${test-exec} PRIVATE -fsycl)
  target_link_options(${test-exec} PRIVATE -fsycl)

  if(ENABLE_CUDA)
    target_compile_options(${test-exec} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(${test-exec} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
  endif()
endforeach()

function(add_shp_test test_name name)
  if (DISABLED_TESTS)
    add_test(NAME ${test_name}-disabled COMMAND ./${name} ${ARGN} --gtest_also_run_disabled_tests)
  else()
    add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  endif()
endfunction()

add_shp_test(shp-3 shp-tests-3 --devicesCount 3)
add_shp_test(shp shp-tests)
