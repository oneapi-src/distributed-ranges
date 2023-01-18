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
