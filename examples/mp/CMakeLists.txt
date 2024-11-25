# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(stencil-slide stencil-slide.cpp)
target_link_libraries(stencil-slide cxxopts DR::mpi)

if(ENABLE_SYCL)
  add_executable(mp_dot_product_benchmark dot_product_benchmark.cpp)
  target_link_libraries(mp_dot_product_benchmark DR::mpi cxxopts)
  add_mp_ctest(
    TEST_NAME mp_dot_product NAME mp_dot_product_benchmark TARGS -n 1000)
endif()

add_executable(vector-add vector-add.cpp)
target_link_libraries(vector-add DR::mpi)
add_mp_ctest(TEST_NAME vector-add NAME vector-add NPROC 2)

function(add_mp_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} cxxopts DR::mpi)
  add_mp_ctest(TEST_NAME ${example_name} NAME ${example_name} NPROC 2)
endfunction()

function(add_mp_example_no_test example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} cxxopts DR::mpi)
endfunction()

add_mp_example(stencil-1d)
add_mp_example(stencil-1d-array)
add_mp_example(stencil-1d-pointer)
add_mp_example(hello_world)
add_mp_example_no_test(sparse_matrix)
add_mp_example_no_test(sparse_benchmark)
add_mp_example_no_test(sparse_matrix_matrix_mul)
add_mp_example_no_test(local_issue)

if(OpenMP_FOUND)
  add_executable(vector-add-ref vector-add-ref.cpp)
  target_link_libraries(vector-add-ref PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                               DR::mpi fmt::fmt)
  add_mp_ctest(TEST_NAME vector-add-ref NAME vector-add-ref NPROC 2)
endif()

add_executable(transpose-ref transpose-ref.cpp)
target_link_libraries(transpose-ref PRIVATE MPI::MPI_CXX MKL::MKL)
