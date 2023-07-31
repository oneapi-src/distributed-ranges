# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  #
  # Google benchmark
  #
  set(BENCHMARK_ENABLE_TESTING off)
  set(BENCHMARK_ENABLE_WERROR off)
  FetchContent_Declare(
    googlebench
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.0)
  FetchContent_MakeAvailable(googlebench)

  # mhp is not under ENABLE_SYCL to check bechmarks also compilation in gcc
  add_subdirectory(mhp)

  if(ENABLE_SYCL)
    add_subdirectory(shp)

    add_custom_target(bench DEPENDS bench-results)
    add_custom_command(
      OUTPUT bench-results
      COMMAND dr-bench analyze -n 1 -n 2 -n 3 -n 4 -s 1000000 -m mhp_gpu -m
              mhp_cpu -m shp -f Stream_ -f Black_Scholes -f DotProduct_
      DEPENDS mhp-bench shp-bench)
  endif()
endif()
