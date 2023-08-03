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

    set(COMMON_PREFIX dr-bench run --vec-size 1000000000)
    set(DR_FILTERS
        --filter Stream_ --filter Black_Scholes --filter Inclusive_Scan_DR
        --filter Reduce_DR)
    set(SERIAL_FILTERS --filter '.*_DPL' --filter '.*_Serial')

    add_custom_target(devcloud-bench DEPENDS devcloud-bench-results)
    add_custom_command(
      OUTPUT devcloud-bench-results
      # devcloud: 4 GPUs, 2 sockets, 112 cores
      COMMAND dr-bench clean
      # DPL is single socket on cpu and single gpu on gpu
      COMMAND ${COMMON_PREFIX} --filter '.*_DPL' --ranks 1 --target mhp_sycl_cpu
              --target mhp_sycl_gpu
      # p2p does not work on devcloud, 1 gpu only
      COMMAND ${COMMON_PREFIX} ${DR_FILTERS} --ranks 1 --target shp_sycl_gpu
      # 1-4 gpu's for mhp sycl
      COMMAND ${COMMON_PREFIX} ${DR_FILTERS} --ranks 1 --ranks 2 --ranks 3
              --ranks 4 --target mhp_sycl_gpu
      # 1 and 2 sockets sycl cpu
      COMMAND ${COMMON_PREFIX} ${DR_FILTERS} --ranks 1 --ranks 2 --target
              mhp_sycl_cpu --target shp_sycl_cpu
      # 1 and 2 sockets direct cpu
      COMMAND ${COMMON_PREFIX} ${DR_FILTERS} --ranks 56 --ranks 112 --target
              mhp_direct_cpu
      COMMAND dr-bench plot
      DEPENDS mhp-bench shp-bench)

    add_custom_target(aurora-bench DEPENDS aurora-bench-results)
    add_custom_command(
      OUTPUT aurora-bench-results
      # aurora: 12 GPUs, 2 sockets, 104 cores
      COMMAND dr-bench clean
      COMMAND ${COMMON_PREFIX} --filter '.*_DPL' --ranks 1 --target mhp_sycl_cpu
              --target shp_sycl_cpu --target mhp_sycl_gpu --target shp_sycl_gpu
      COMMAND
        ${COMMON_PREFIX} ${DR_FILTERS} --no-fork --ranks 1 --ranks 2 --ranks 3
        --ranks 4 --ranks 5 --ranks 6 --ranks 7 --ranks 8 --ranks 9 --ranks 10
        --ranks 11 --ranks 12 --target mhp_sycl_gpu --target shp_sycl_gpu
      COMMAND ${COMMON_PREFIX} ${DR_FILTERS} --no-fork --ranks 1 --ranks 2
              --target mhp_sycl_cpu --target shp_sycl_cpu
      COMMAND ${COMMON_PREFIX} ${DR_FILTERS} --no-fork --ranks 52 --ranks 104
              --target mhp_direct_cpu
      COMMAND dr-bench plot
      DEPENDS mhp-bench shp-bench)

  endif()
endif()
