# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(ENABLE_CUDA)
  # because sort.cpp compilation fails with
  # dpl/pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort_one_wg.h warning:
  # attribute argument 16 is invalid and will be ignored; CUDA requires
  # sub_group size 32
  add_compile_options(-Wno-error=cuda-compat)
endif()

# cmake-format: off
add_executable(
  shp-bench
  shp-bench.cpp
  gemm.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
  ../common/inclusive_exclusive_scan.cpp
  ../common/sort.cpp
  ../common/stream.cpp)
# cmake-format: on

if(NOT ENABLE_CUDA)
  # black_scholes.cpp does not compile in CUDA because it uses std::log
  target_sources(shp-bench PRIVATE ../common/black_scholes.cpp)
endif()

target_compile_definitions(shp-bench PRIVATE BENCH_SHP)
target_link_libraries(shp-bench benchmark::benchmark cxxopts DR::shp)

add_shp_ctest(shp-bench shp-bench --vector-size 20000)
