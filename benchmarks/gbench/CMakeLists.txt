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

    add_custom_target(bench DEPENDS bench-gpu-results bench-cpu-results)
    add_custom_command(
      OUTPUT bench-gpu-results
      COMMAND dr-bench --analysis_id gpu analyze -n 1 -n 2 -n 3 -n 4 -s 1000000 -m mhp_gpu_sycl -m
              shp_gpu_sycl
      DEPENDS mhp-bench shp-bench)

    add_custom_command(
      OUTPUT bench-cpu-results
      COMMAND dr-bench --analysis_id cpu analyze --no-plot -n 1 -n 2 -s 1000000 -m mhp_cpu_sycl -m shp_cpu_sycl
      COMMAND dr-bench --analysis_id cpu analyze --no-plot -n 56 -n 112 -s 1000000 -m mhp_cpu_direct
      COMMAND dr-bench --analysis_id cpu plot
      DEPENDS mhp-bench shp-bench)
  endif()
endif()
