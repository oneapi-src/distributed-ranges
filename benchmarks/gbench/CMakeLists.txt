# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  #
  # Google benchmark
  #
  set(BENCHMARK_ENABLE_TESTING off)
  set(BENCHMARK_ENABLE_WERROR off)
  # oneApi 2024.1 compiler refuses to compile code with standard less than C++17
  set(BENCHMARK_CXX_STANDARD 20)
  FetchContent_Declare(
    googlebench
    GIT_REPOSITORY https://github.com/lslusarczyk/benchmark.git
    GIT_TAG set-cxx-std)
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

    add_custom_target(jfcst-bench DEPENDS jfcst-bench-results)
    add_custom_command(
      OUTPUT jfcst-bench-results
      # github actions log annotation
      COMMAND echo "::endgroup::"
      COMMAND dr-bench clean
      COMMAND dr-bench suite --gpus 4 --ppn 2
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    add_custom_target(aurora-bench-1 DEPENDS aurora-bench-results)
    add_custom_command(
      OUTPUT aurora-bench-results
      # github actions log annotation
      COMMAND echo "::endgroup::"
      COMMAND dr-bench clean
      COMMAND dr-bench suite --gpus 12 --ppn 12
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    # shp fails with PI_OUT_OF_RESOURCES when GPUS>4, 3DFFT fails always on
    # 2024.1 and sometimes on 2023.2 other shp benchmark also failed in 2024 on
    # 8 GPUs in initialization
    add_custom_target(aurora-bench-1-shplimited
                      DEPENDS aurora-bench-results-shplimited)
    add_custom_command(
      OUTPUT aurora-bench-results-shplimited
      # github actions log annotation
      COMMAND echo "::endgroup::"
      COMMAND dr-bench clean
      COMMAND dr-bench suite --gpus 12 --ppn 12 --mhp-only
      COMMAND dr-bench suite --gpus 12 --ppn 12 --reference-only
      COMMAND dr-bench suite --gpus 4 --ppn 12 --shp-only
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    add_custom_target(aurora-bench-2 DEPENDS aurora-bench-results-2)
    add_custom_command(
      OUTPUT aurora-bench-results-2
      # github actions log annotation
      COMMAND echo "::endgroup::"
      COMMAND dr-bench clean
      COMMAND dr-bench suite --min-gpus 13 --gpus 24 --ppn 12 --mhp-only
      COMMAND dr-bench suite --gpus 12 --ppn 12
      COMMAND dr-bench plot
      DEPENDS xhp-bench)

    add_custom_target(aurora-bench-plus-2 DEPENDS aurora-bench-results-plus-2)
    add_custom_command(
      OUTPUT aurora-bench-results-plus-2
      # github actions log annotation
      COMMAND echo "::endgroup::"
      COMMAND dr-bench suite --min-gpus 13 --gpus 24 --ppn 12 --mhp-only
      COMMAND dr-bench plot
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
