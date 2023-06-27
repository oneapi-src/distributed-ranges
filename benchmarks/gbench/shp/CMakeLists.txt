# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  shp-bench
  shp-bench.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
  ../common/stream.cpp
  ../common/black_scholes.cpp
)
target_compile_definitions(shp-bench PRIVATE BENCH_SHP)
target_link_libraries(
  shp-bench
  benchmark::benchmark
  cxxopts
  DR::mpi
)

add_test(NAME shp-bench COMMAND ./shp-bench --vector-size 20000)
