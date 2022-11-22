add_executable(
  cpu-tests
  cpu-tests.cpp
  communicator.cpp
  remote_memory.cpp
  remote_vector.cpp
  distributed_vector.cpp
  unstructured_halo.cpp
  mdspan.cpp
)

target_link_libraries(
  cpu-tests
  GTest::gtest_main
  cxxopts
  DR::mpi
)

add_executable(
  cpu-tests-4
  cpu-tests.cpp
  span_halo.cpp
)

target_link_libraries(
  cpu-tests-4
  GTest::gtest_main
  cxxopts
  DR::mpi
)

add_mpi_test(cpu-mpi-2 cpu-tests 2)
add_mpi_test(cpu-mpi-4 cpu-tests-4 4)
