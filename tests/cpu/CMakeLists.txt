add_executable(
  cpu-tests
  cpu-tests.cpp
  communicator.cpp
  remote_memory.cpp
  remote_vector.cpp
  distributed_vector.cpp
  halo.cpp
  mdspan.cpp
)

target_link_libraries(
  cpu-tests
  GTest::gtest_main
  DR::dr
)

add_mpi_test(cpu-mpi-2 cpu-tests 2)
