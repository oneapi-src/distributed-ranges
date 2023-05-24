# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(stencil-slide stencil-slide.cpp)
target_link_libraries(stencil-slide cxxopts DR::mpi)

if(ENABLE_SYCL)
  add_executable(mhp_dot_product_benchmark dot_product_benchmark.cpp)
  target_compile_options(mhp_dot_product_benchmark PRIVATE -fsycl)
  target_link_libraries(mhp_dot_product_benchmark DR::mpi cxxopts)
  add_mpi_test(mhp_dot_product mhp_dot_product_benchmark 2)
endif()

add_executable(vector-add vector-add.cpp)
target_link_libraries(vector-add DR::mpi)
add_mpi_test(vector-add vector-add 2)

add_executable(stencil-1d stencil-1d.cpp)
target_link_libraries(
  stencil-1d
  cxxopts
  DR::mpi)
add_mpi_test(stencil-1d stencil-1d 2)

add_executable(tile-example tile-example.cpp)
target_link_libraries(
  tile-example
  cxxopts
  DR::mpi)
add_mpi_test(tile-example tile-example 2)

add_executable(stencil-1d-array stencil-1d-array.cpp)
target_link_libraries(
  stencil-1d-array
  cxxopts
  DR::mpi)
add_mpi_test(stencil-1d-array stencil-1d-array 2)

add_executable(stencil-2d-matrix stencil-2d-matrix.cpp)
target_link_libraries(
  stencil-2d-matrix
  cxxopts
  DR::mpi)
add_mpi_test(stencil-2d-matrix stencil-2d-matrix 2)

add_executable(stencil-2d-matrix-rows stencil-2d-matrix-rows.cpp)
target_link_libraries(
  stencil-2d-matrix-rows
  cxxopts
  DR::mpi)
add_mpi_test(stencil-2d-matrix-rows stencil-2d-matrix-rows 2)

add_executable(wave_equation wave_equation.cpp)
target_link_libraries(
  wave_equation
  cxxopts
  DR::mpi)
add_mpi_test(wave_equation wave_equation 4)

if(OpenMP_FOUND)
  add_executable(vector-add-ref vector-add-ref.cpp)
  target_link_libraries(vector-add-ref PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX fmt::fmt)
  add_mpi_test(vector-add-ref vector-add-ref 2)
endif()

add_executable(transpose-ref transpose-ref.cpp)
target_link_libraries(transpose-ref PRIVATE MPI::MPI_CXX MKL::MKL)
