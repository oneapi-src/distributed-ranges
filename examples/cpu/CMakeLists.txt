#add_executable(vector-add-cpu vector-add-cpu.cpp)
#target_link_libraries(vector-add-cpu DR::dr)
#add_mpi_test(vector-add-cpu vector-add-cpu 2)

add_executable(vector-add-cpu-ref vector-add-cpu-ref.cpp)
target_link_libraries(vector-add-cpu-ref PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
add_mpi_test(vector-add-cpu-ref vector-add-cpu-ref 2)
