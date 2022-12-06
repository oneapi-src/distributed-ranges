# only tested with 2 ranks
add_executable(
  cpu-tests-2
  cpu-tests.cpp
  communicator.cpp
  remote_memory.cpp
  remote_vector.cpp
  distributed_vector.cpp
  unstructured_halo.cpp
  mdspan.cpp
)

target_link_libraries(
  cpu-tests-2
  GTest::gtest_main
  cxxopts
  DR::mpi
)

# tested with a variable number of ranks
add_executable(
  cpu-tests-n

  cpu-tests.cpp
  fill.cpp
  reduce.cpp
  span_halo.cpp
  transform.cpp
  transform_reduce.cpp
)

target_link_libraries(
  cpu-tests-n
  GTest::gtest_main
  cxxopts
  DR::mpi
)

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

add_mpi_test(cpu-mpi cpu-tests-2 2)

if (NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1
  add_mpi_test(cpu-mpi-1 cpu-tests-n 1)
endif()
add_mpi_test(cpu-mpi-2 cpu-tests-n 2)
add_mpi_test(cpu-mpi-3 cpu-tests-n 3)
add_mpi_test(cpu-mpi-4 cpu-tests-n 4)
