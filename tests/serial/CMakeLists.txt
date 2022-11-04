add_executable(
  serial-tests
  serial-tests.cpp
)

target_link_libraries(
  serial-tests
  GTest::gtest_main
)

add_test(NAME serial-tests COMMAND ./serial-tests)
