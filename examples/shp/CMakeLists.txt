# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

function(add_shp_example_no_test example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} DR::shp)
  target_compile_options(${example_name} PUBLIC -fsycl)
  target_link_options(${example_name} PUBLIC -fsycl)
  if(ENABLE_CUDA)
    target_compile_options(${example_name} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(${example_name} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
  endif()
endfunction()

function(add_shp_example example_name)
  add_shp_example_no_test(${example_name})
  add_test(NAME ${example_name} COMMAND ./${example_name})
endfunction()

add_shp_example(vector_example)
add_shp_example(test_range)
add_shp_example(dot_product)
add_shp_example(dot_product_benchmark)
add_shp_example(inclusive_scan_example)
# unsatisfied dependency of grb/grb.hpp
# add_shp_example(gemv_benchmark)
add_shp_example_no_test(copy_test)
add_shp_example_no_test(gather_test)
add_shp_example_no_test(gemv_example)  # PI_ERROR_DEVICE_NOT_FOUND
add_shp_example_no_test(matrix_example)  # PI_ERROR_DEVICE_NOT_FOUND
add_shp_example(sparse_test)
add_shp_example_no_test(take_example)  # PI_ERROR_DEVICE_NOT_FOUND
add_shp_example_no_test(zip_example)  # PI_ERROR_DEVICE_NOT_FOUND
