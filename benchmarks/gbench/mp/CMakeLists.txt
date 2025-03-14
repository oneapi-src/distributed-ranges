# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# cmake-format: off
add_executable(
  mp-bench
  mp-bench.cpp
  ../common/distributed_vector.cpp
  ../common/dot_product.cpp
  ../common/inclusive_exclusive_scan.cpp
  ../common/sort.cpp
  ../common/stream.cpp
  streammp.cpp
  rooted.cpp
  gemv.cpp
  stencil_1d.cpp
  stencil_2d.cpp
  chunk.cpp
  # mdspan.cpp
  mpi.cpp)
# cmake-format: on

# disabled with SYCL due to DRA-135
if(NOT ENABLE_SYCL)
  target_sources(mp-bench PRIVATE mdspan.cpp)
endif()

if(ENABLE_SYCL)
  target_sources(mp-bench PRIVATE fft3d.cpp)
endif()

if(NOT ENABLE_CUDA)
  # does not compile in CUDA because: black_scholes.cpp uses std::log
  # shallow_water, wave_equation uses uses exp
  target_sources(mp-bench PRIVATE ../common/black_scholes.cpp shallow_water.cpp
                                  wave_equation.cpp)
endif()

# mp-quick-bench is for development. By reducing the number of source files, it
# builds much faster. Change the source files to match what you need to test. It
# is OK to commit changes to the source file list.
add_executable(mp-quick-bench mp-bench.cpp gemv.cpp)

foreach(mp-bench-exec IN ITEMS mp-bench mp-quick-bench)
  target_compile_definitions(${mp-bench-exec} PRIVATE BENCH_MP)
  target_link_libraries(${mp-bench-exec} benchmark::benchmark cxxopts DR::mpi)
  if(ENABLE_ISHMEM)
    target_link_ishmem(${mp-bench-exec})
  endif()
  if(ENABLE_SYCL)
    target_link_libraries(${mp-bench-exec} MKL::MKL_DPCPP)
  endif()
endforeach()

if(ENABLE_SYCL)
  # target_sources(mp-quick-bench PRIVATE fft3d.cpp)
endif()

cmake_path(GET MPI_CXX_ADDITIONAL_INCLUDE_DIRS FILENAME MPI_IMPL)

# debug mp-bench is too slow
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT MPI_IMPL STREQUAL "openmpi")
  # MPI_Win_create fails for communicator with size 1 30000 is minimum because
  # of static column size for stencil2D disable DPL benchmarks because we get
  # intermittent fails with: ONEAPI_DEVICE_SELECTOR=opencl:cpu mpirun -n 1
  # ./mp-bench --vector-size 30000 --rows 100 --columns 100 --check
  add_mp_ctest(
    NAME mp-bench TIMEOUT 200 TARGS --vector-size 30000 --rows 100 --columns
    100 --check --benchmark_filter=-FFT3D.*)
  if(ENABLE_SYCL)
    add_mp_ctest(
      NAME mp-bench TIMEOUT 200 SYCL TARGS --vector-size 30000 --rows 100
      --columns 100 --check --benchmark_filter=-.*DPL.*)
  endif()
endif()

add_executable(wave_equation wave_equation.cpp)
target_link_libraries(wave_equation cxxopts DR::mpi)
target_compile_definitions(wave_equation PRIVATE STANDALONE_BENCHMARK)
add_mp_ctest(NAME wave_equation)
# add_mp_ctest(TEST_NAME wave_equation_fused NAME wave_equation TARGS -f) #
# DRA-92
if(ENABLE_SYCL)
  add_mp_ctest(
    TEST_NAME wave_equation-sycl NAME wave_equation NPROC 2 SYCL)
  add_mp_ctest(
    TEST_NAME wave_equation_fused-sycl NAME wave_equation NPROC 2 SYCL TARGS -f)
endif()

add_executable(shallow_water shallow_water.cpp)
target_link_libraries(shallow_water cxxopts DR::mpi)
target_compile_definitions(shallow_water PRIVATE STANDALONE_BENCHMARK)

# issue DRA-23 add_mp_ctest(TEST_NAME shallow_water NAME shallow_water NPROC 1)
# add_mp_ctest( TEST_NAME shallow_water_fused NAME shallow_water NPROC 1 TARGS
# -f)

if(ENABLE_SYCL)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # too long (or hangs?) in debug

    add_mp_ctest(
      TEST_NAME shallow_water-sycl NAME shallow_water NPROC 2 SYCL)
    add_mp_ctest(
      TEST_NAME shallow_water_fused-sycl NAME shallow_water NPROC 2 SYCL TARGS
      -f)
  endif()

  add_executable(fft3d_mp fft3d.cpp)
  target_link_libraries(fft3d_mp cxxopts DR::mpi MKL::MKL_DPCPP)
  target_compile_definitions(fft3d_mp PRIVATE STANDALONE_BENCHMARK)
  add_mp_ctest(TEST_NAME fft3d_mp NAME fft3d_mp NPROC 2)
endif()
