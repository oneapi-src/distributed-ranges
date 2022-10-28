include(GoogleTest)

function(add_mpi_test test_name name processes)
  add_test(NAME ${test_name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes} ${MPIEXEC_PREFLAGS} ./${name} ${ARGN} COMMAND_EXPAND_LISTS)
endfunction()

add_executable(
  cpu-tests
  cpu-tests.cpp
)
target_link_libraries(
  cpu-tests
  GTest::gtest_main
  DR::dr
)

add_executable(
  cpu-mpi-tests
  cpu-mpi-tests.cpp
  communicator.cpp
  remote_memory.cpp
  remote_vector.cpp
  distributed_vector.cpp
  halo.cpp
)
target_link_libraries(
  cpu-mpi-tests
  GTest::gtest_main
  DR::dr
)

gtest_discover_tests(cpu-tests cpu-mpi-tests)
add_mpi_test(cpu-mpi-2 cpu-mpi-tests 2)
