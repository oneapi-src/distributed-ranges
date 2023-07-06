# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  shp-bench
  shp-bench.cpp ../common/distributed_vector.cpp ../common/dot_product.cpp
  ../common/stream.cpp ../common/black_scholes.cpp)
target_compile_definitions(shp-bench PRIVATE BENCH_SHP)
target_link_libraries(shp-bench benchmark::benchmark cxxopts DR::mpi)

target_link_libraries(
  shp-bench-only-fsycl
  benchmark::benchmark
  cxxopts
  DR::mpi
)

target_compile_options(shp-bench PRIVATE -O3 -qopenmp -qopt-streaming-stores=always -xCORE-AVX512 -qopt-zmm-usage=high)
target_compile_options(shp-bench-only-fsycl PRIVATE -O3)

add_test(NAME shp-bench COMMAND ./shp-bench --vector-size 20000)
add_test(NAME shp-bench-only-fsycl COMMAND ./shp-bench-only-fsycl --vector-size 20000)
