# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  shp-bench
  shp-bench.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
)
target_link_libraries(
  shp-bench
  benchmark::benchmark
  cxxopts
  DR::mpi
)

target_compile_options(shp-bench PRIVATE -fsycl)
add_test(NAME shp-bench COMMAND ./shp-bench)
