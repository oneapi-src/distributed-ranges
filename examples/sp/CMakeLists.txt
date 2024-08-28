# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

function(add_sp_example_no_test example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} DR::sp)
endfunction()

function(add_sp_example example_name)
  add_sp_example_no_test(${example_name})
  add_sp_ctest(${example_name} ${example_name} ${ARGN})
endfunction()

add_sp_example(vector_example)
add_sp_example(dot_product)
add_sp_example(sort)
add_sp_example_no_test(sort_benchmark)
add_sp_example(inclusive_scan_example)
add_sp_example(exclusive_scan_example)
# unsatisfied dependency of grb/grb.hpp add_sp_example(gemv_benchmark)
add_sp_example_no_test(dot_product_benchmark)
add_sp_example_no_test(inclusive_scan_benchmark)
add_sp_example_no_test(exclusive_scan_benchmark)
add_sp_example_no_test(copy_test)
add_sp_example_no_test(gather_test)
add_sp_example_no_test(gemv_example) # PI_ERROR_DEVICE_NOT_FOUND
add_sp_example_no_test(matrix_example) # PI_ERROR_DEVICE_NOT_FOUND
add_sp_example_no_test(gemm_example) # PI_ERROR_DEVICE_NOT_FOUND
add_sp_example_no_test(gemm_benchmark) # PI_ERROR_DEVICE_NOT_FOUND
add_sp_example(sparse_test)
add_sp_example_no_test(take_example) # PI_ERROR_DEVICE_NOT_FOUND
add_sp_example_no_test(test_range) # due to DRA-81
add_sp_example_no_test(zip_example) # PI_ERROR_DEVICE_NOT_FOUND
add_sp_example_no_test(black_scholes_benchmark) # PI_ERROR_DEVICE_NOT_FOUND
