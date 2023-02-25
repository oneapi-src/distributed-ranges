# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

function(add_shp_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} DR::shp)
  target_compile_options(${example_name} PUBLIC -fsycl)
  target_link_options(${example_name} PUBLIC -fsycl)
  if(ENABLE_CUDA)
    target_compile_options(${example_name} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
    target_link_options(${example_name} PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
  endif()
  add_test(NAME ${example_name} COMMAND ./${example_name})
endfunction()

add_shp_example(vector_example)
add_shp_example(test_range)
add_shp_example(inclusive_scan_example)
