# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# tested with a variable number of ranks
add_executable(
  cpu-fuzz
  cpu-fuzz.cpp

  algorithms.cpp
)

target_compile_options(cpu-fuzz PUBLIC -fsanitize=fuzzer)
target_link_options(cpu-fuzz PUBLIC -fsanitize=fuzzer)

target_link_libraries(
  cpu-fuzz
  cxxopts
  DR::mpi
)
function(add_long_fuzz_mpi_test test_name name processes)
  add_test(NAME ${test_name} CONFIGURATIONS long_fuzz COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes} ${MPIEXEC_PREFLAGS} ./${name} ${ARGN} COMMAND_EXPAND_LISTS)
endfunction()

add_mpi_test(cpu-fuzz-commit cpu-fuzz 1 -max_len=16 -runs=1000000 -ignore_remaining_args=1 -controller=1)

add_long_fuzz_mpi_test(cpu-fuzz-4-0 cpu-fuzz 4 -max_len=16 -runs=10000000 -ignore_remaining_args=1 -controller=0)
add_long_fuzz_mpi_test(cpu-fuzz-4-1 cpu-fuzz 4 -max_len=16 -runs=10000000 -ignore_remaining_args=1 -controller=1)
add_long_fuzz_mpi_test(cpu-fuzz-4-2 cpu-fuzz 4 -max_len=16 -runs=10000000 -ignore_remaining_args=1 -controller=2)
add_long_fuzz_mpi_test(cpu-fuzz-4-3 cpu-fuzz 4 -max_len=16 -runs=10000000 -ignore_remaining_args=1 -controller=3)
