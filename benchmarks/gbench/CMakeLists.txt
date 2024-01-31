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

  if(ENABLE_CUDA)
    # because sort.cpp compilation fails with
    # dpl/pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort_one_wg.h warning:
    # attribute argument 16 is invalid and will be ignored; CUDA requires
    # sub_group size 32
    add_compile_options(-Wno-error=cuda-compat)
  endif()

  # mhp is not under ENABLE_SYCL to check benchmarks also compilation in gcc
  add_subdirectory(mhp)

  if(ENABLE_SYCL)
    add_subdirectory(shp)

    add_custom_target(xhp-bench DEPENDS mhp-bench shp-bench)

    add_custom_target(devcloud-bench DEPENDS devcloud-bench-results)
    add_custom_command(
      OUTPUT devcloud-bench-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --gpus 4 --sockets 2 --cores-per-socket 56
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    add_custom_target(aurora-bench DEPENDS aurora-bench-results)
    add_custom_command(
      OUTPUT aurora-bench-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --gpus 12
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    add_custom_target(aurora-bench-2 DEPENDS aurora-bench-2-results)
    add_custom_command(
      OUTPUT aurora-bench-2-results
      COMMAND dr-bench clean --prefix aurora-2
      # speedup reference relative to 1 gpu
      COMMAND dr-bench suite --prefix aurora-2 --gpus 1
      # measurements
      COMMAND dr-bench suite --prefix aurora-2 --nodes 2 --gpus 12
      COMMAND dr-bench plot --prefix aurora-2
      DEPENDS xhp-bench)

    add_custom_target(aurora-bench-4 DEPENDS aurora-bench-4-results)
    add_custom_command(
      OUTPUT aurora-bench-4-results
      COMMAND dr-bench clean --prefix aurora-4
      COMMAND dr-bench suite --prefix aurora-4 --nodes 4 --gpus 12
      COMMAND dr-bench plot --prefix aurora-4
      DEPENDS xhp-bench)

    add_custom_target(quick-bench-gpu DEPENDS quick-bench-gpu-results)
    add_custom_command(
      OUTPUT quick-bench-gpu-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --reps 10 --gpus 2
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    add_custom_target(quick-bench-cpu DEPENDS quick-bench-cpu-results)
    add_custom_command(
      OUTPUT quick-bench-cpu-results
      COMMAND dr-bench clean
      COMMAND dr-bench suite --reps 10 --cores-per-socket 4
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

  endif()
endif()
