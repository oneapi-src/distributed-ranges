# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# cmake-format: off
add_executable(
  mhp-bench
  mhp-bench.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
  ../common/inclusive_exclusive_scan.cpp
  ../common/sort.cpp
  ../common/stream.cpp
  rooted.cpp
  stencil_1d.cpp
  stencil_2d.cpp
  chunk.cpp
  mdspan.cpp
  mpi.cpp)
# cmake-format: on

if(NOT ENABLE_CUDA)
  # does not compile in CUDA because: black_scholes.cpp uses std::log
  # shallow_water, wave_equation uses uses exp
  target_sources(mhp-bench PRIVATE ../common/black_scholes.cpp
                                   shallow_water.cpp wave_equation.cpp)
endif()

# mhp-quick-bench is for development. By reducing the number of source files, it
# builds much faster. Change the source files to match what you need to test. It
# is OK to commit changes to the source file list.
add_executable(mhp-quick-bench mhp-bench.cpp)

foreach(mhp-bench-exec IN ITEMS mhp-bench mhp-quick-bench)
  target_compile_definitions(${mhp-bench-exec} PRIVATE BENCH_MHP)
  target_link_libraries(${mhp-bench-exec} benchmark::benchmark cxxopts DR::mpi)
  target_link_ishmem(${mhp-bench-exec})
endforeach()

if(ENABLE_SYCL)
  target_sources(mhp-quick-bench PRIVATE transpose.cpp)
  target_link_libraries(mhp-quick-bench benchmark::benchmark cxxopts DR::mpi
                        MKL::MKL_DPCPP)
endif()

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

# debug mhp-bench is too slow
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1 30000 is minimum because
  # of static column size for stencil2D disable DPL benchmarks because we get
  # intermittent fails with: ONEAPI_DEVICE_SELECTOR=opencl:cpu mpirun -n 1
  # ./mhp-bench --vector-size 30000 --rows 100 --columns 100 --check
  add_mhp_ctest(
    NAME mhp-bench TARGS --vector-size 30000 --rows 100 --columns 100 --check)
  if(ENABLE_SYCL)
    add_mhp_ctest(
      NAME mhp-bench SYCL TARGS --vector-size 30000 --rows 100 --columns 100
      --check --benchmark_filter=-.*DPL.*)
  endif()
endif()

add_executable(wave_equation wave_equation.cpp)
target_link_libraries(wave_equation cxxopts DR::mpi)
target_compile_definitions(wave_equation PRIVATE STANDALONE_BENCHMARK)
add_mhp_ctest(NAME wave_equation)
add_mhp_ctest(TEST_NAME wave_equation_fused NAME wave_equation TARGS -f)
if(ENABLE_SYCL)
  add_mhp_ctest(
    TEST_NAME wave_equation-sycl NAME wave_equation NPROC 2 SYCL)
  add_mhp_ctest(
    TEST_NAME wave_equation_fused-sycl NAME wave_equation NPROC 2 SYCL TARGS -f)
endif()

add_executable(shallow_water shallow_water.cpp)
target_link_libraries(shallow_water cxxopts DR::mpi)
target_compile_definitions(shallow_water PRIVATE STANDALONE_BENCHMARK)
# temporarily disabled - issue DRA-23 add_mhp_ctest(shallow_water shallow_water
# 1) add_mhp_ctest(shallow_water_fused shallow_water 1 -f)
if(ENABLE_SYCL)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # too long (or hangs?) in debug

    add_mhp_ctest(
      TEST_NAME shallow_water-sycl NAME shallow_water NPROC 2 SYCL)
    add_mhp_ctest(
      TEST_NAME shallow_water_fused-sycl NAME shallow_water NPROC 2 SYCL TARGS
      -f)
  endif()

  add_executable(fft3d_mhp fft3d.cpp)
  target_link_libraries(fft3d_mhp cxxopts DR::mpi MKL::MKL_DPCPP)
  target_compile_definitions(fft3d_mhp PRIVATE STANDALONE_BENCHMARK)
  add_mhp_ctest(TEST_NAME fft3d_mhp NAME fft3d_mhp NPROC 2)
endif()
