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

    set(COMMON_PREFIX
        dr-bench --analysis_id daily analyze --no-plot -s 1000000 -f Stream_ -f
        Black_Scholes -f DotProduct_)
    message(STATUS "Common prefix ${COMMON_PREFIX}")
    add_custom_target(bench DEPENDS bench-results)
    add_custom_command(
      OUTPUT bench-results
      # devcloud: 4 GPUs, 2 sockets, 112 cores
      COMMAND ${COMMON_PREFIX} -n 1 -n 2 -n 3 -n 4 -t mhp_sycl_gpu -t
              shp_sycl_gpu
      COMMAND ${COMMON_PREFIX} -n 1 -n 2 -m mhp_sycl_cpu -m shp_sycl_cpu
      COMMAND ${COMMON_PREFIX} -n 1 -n 2 -n 4 -n 8 -n 56 -n 112 -t
              mhp_direct_cpu
      COMMAND dr-bench --analysis_id daily plot
      DEPENDS mhp-bench shp-bench)
  endif()
endif()
