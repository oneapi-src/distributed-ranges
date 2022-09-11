add_library(dr INTERFACE)
add_library(DR::dr ALIAS dr)

find_package(MPI REQUIRED)
target_link_libraries(dr INTERFACE MPI::MPI_CXX)
