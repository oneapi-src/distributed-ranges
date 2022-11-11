function(add_shp_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} DR::shp)
  add_test(NAME ${example_name} COMMAND .${example_name})
endfunction()

add_shp_example(vector_example)
add_shp_example(test_range)
