# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# cmake-format: off
add_executable(
  shp-bench
  shp-bench.cpp
  ../common/sort.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
  ../common/inclusive_scan.cpp
  ../common/stream.cpp)
# cmake-format: on

if(NOT ENABLE_CUDA)
  # black_scholes.cpp does not compile in CUDA because it uses std::log
  target_sources(shp-bench PRIVATE ../common/black_scholes.cpp)
endif()

target_compile_definitions(shp-bench PRIVATE BENCH_SHP)
target_link_libraries(shp-bench benchmark::benchmark cxxopts DR::mpi)

add_test(NAME shp-bench COMMAND ./shp-bench --vector-size 20000)
