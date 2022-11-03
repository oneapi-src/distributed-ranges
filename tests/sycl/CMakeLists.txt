add_executable(
  sycl-tests
  sycl-tests.cpp
)

target_link_libraries(
  sycl-tests
  GTest::gtest_main
  DR::dr
)

target_compile_options(
  sycl-tests
  PRIVATE
  -fsycl)
