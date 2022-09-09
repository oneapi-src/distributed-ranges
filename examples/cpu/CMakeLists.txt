add_executable(vector-add vector-add.cpp)
target_link_libraries(vector-add DR::dr)
add_test(NAME vector-add COMMAND ./vector-add COMMAND_EXPAND_LISTS)
