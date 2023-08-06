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

    add_custom_target(devcloud-bench DEPENDS devcloud-bench-results)
    add_custom_command(
      OUTPUT devcloud-bench-results
      COMMAND dr-bench clean
      # devcloud does not have p2p, so only 1 gpu for shp
      COMMAND dr-bench suite --no-p2p --gpus 4 --cores-per-socket 56
      COMMAND dr-bench plot
      DEPENDS mhp-bench shp-bench)

    add_custom_target(aurora-bench DEPENDS aurora-bench-results)
    add_custom_command(
      OUTPUT aurora-bench-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --gpus 12
      COMMAND dr-bench plot
      DEPENDS mhp-bench shp-bench)

    add_custom_target(quick-bench-gpu DEPENDS quick-bench-gpu-results)
    add_custom_command(
      OUTPUT quick-bench-gpu-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --reps 10 --gpus 2
      COMMAND dr-bench plot
      DEPENDS mhp-bench shp-bench)

    add_custom_target(quick-bench-cpu DEPENDS quick-bench-cpu-results)
    add_custom_command(
      OUTPUT quick-bench-cpu-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --reps 10 --cores-per-socket 4
      COMMAND dr-bench plot
      DEPENDS mhp-bench shp-bench)

  endif()
endif()
