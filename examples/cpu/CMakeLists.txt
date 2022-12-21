# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(vector-add-cpu vector-add-cpu.cpp)
target_link_libraries(vector-add-cpu DR::mpi)
add_mpi_test(vector-add-cpu vector-add-cpu 2)

add_executable(vector-add-presentation vector-add-presentation.cpp)
target_link_libraries(vector-add-presentation DR::mpi)
add_mpi_test(vector-add-presentation vector-add-presentation 2)

add_executable(stencil-1d-cpu stencil-1d-cpu.cpp)
target_link_libraries(
  stencil-1d-cpu
  cxxopts
  DR::mpi)
add_mpi_test(stencil-1d-cpu stencil-1d-cpu 2)

if(OpenMP_FOUND)
  add_executable(vector-add-cpu-ref vector-add-cpu-ref.cpp)
  target_link_libraries(vector-add-cpu-ref PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX fmt::fmt)
  add_mpi_test(vector-add-cpu-ref vector-add-cpu-ref 2)
endif()

add_executable(transpose-cpu-ref transpose-cpu-ref.cpp)
target_link_libraries(transpose-cpu-ref PRIVATE MPI::MPI_CXX MKL::MKL)

#add_executable(transpose-cpu transpose-cpu.cpp)
#target_link_libraries(transpose-cpu DR::mpi MKL::MKL)
#add_mpi_test(transpose-cpu transpose-cpu 2)
