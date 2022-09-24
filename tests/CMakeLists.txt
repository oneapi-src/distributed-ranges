add_executable(
  cpu-tests
  cpu-tests.cpp
)
target_link_libraries(
  cpu-tests
  GTest::gtest_main
  DR::dr
)

include(GoogleTest)
gtest_discover_tests(cpu-tests)
