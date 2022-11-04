include(GoogleTest)

function(add_mpi_test test_name name processes)
  add_test(NAME ${test_name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes} ${MPIEXEC_PREFLAGS} ./${name} ${ARGN} COMMAND_EXPAND_LISTS)
endfunction()

add_subdirectory(serial)
add_subdirectory(cpu)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  add_subdirectory(sycl)
endif()
