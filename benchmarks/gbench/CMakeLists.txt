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

  # mp is not under ENABLE_SYCL to check benchmarks also compilation in gcc
  add_subdirectory(mp)

  if(ENABLE_SYCL)
    add_subdirectory(sp)
    add_custom_target(xp-bench DEPENDS mp-bench sp-bench)
  endif()
endif()
