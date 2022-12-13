add_executable(
  sycl-mpi-tests
  sycl-mpi-tests.cpp
  distributed_vector.cpp
  dpl.cpp
  reduce.cpp
  transform.cpp
  unstructured_halo.cpp
)

target_link_libraries(
  sycl-mpi-tests
  GTest::gtest_main
  cxxopts
  DR::mpi
)
target_compile_options(sycl-mpi-tests PRIVATE -fsycl)
target_link_options(sycl-mpi-tests PRIVATE -fsycl)
if(ENABLE_CUDA)
  target_compile_options(sycl-mpi-tests PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
  target_link_options(sycl-mpi-tests PUBLIC -fsycl-targets=nvptx64-nvidia-cuda)
endif()

add_mpi_test(sycl-mpi-1 sycl-mpi-tests 1)
add_mpi_test(sycl-mpi-2 sycl-mpi-tests 2)
add_mpi_test(sycl-mpi-3 sycl-mpi-tests 3)
add_mpi_test(sycl-mpi-4 sycl-mpi-tests 4)
