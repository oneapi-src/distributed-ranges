add_executable(vector-add-serial vector-add-serial.cpp)
add_test(vector-add-serial ./vector-add-serial)

add_executable(transpose-serial transpose-serial.cpp)
target_link_libraries(transpose-serial PRIVATE MKL::MKL)
add_test(transpose-serial ./transpose-serial)
