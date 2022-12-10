add_executable(
  sycl-mpi-tests
  sycl-mpi-tests.cpp
  distributed_vector.cpp
  dpl.cpp
  reduce.cpp
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

add_test(NAME sycl-mpi COMMAND ./sycl-mpi-tests)
