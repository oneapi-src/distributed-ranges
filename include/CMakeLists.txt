# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

FetchContent_Declare(
  dpl
  GIT_REPOSITORY https://github.com/rscohn2/oneDPL.git
  GIT_TAG patch-1
  )
FetchContent_MakeAvailable(dpl)

# mpi:  multi-process

add_library(dr_mpi INTERFACE)
add_library(DR::mpi ALIAS dr_mpi)

find_package(MPI REQUIRED)
find_package(MKL REQUIRED)
find_package(TBB REQUIRED)
target_link_libraries(dr_mpi INTERFACE MPI::MPI_CXX range-v3 std::mdspan MKL::MKL TBB::tbb)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION STRLESS "11")
  # gcc 10 uses TBB API that were removed
  message(NOTICE "Disabling PSTL TBB parallel backend for gcc 10")
  target_compile_definitions(dr_mpi INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
endif()
if (ENABLE_FORMAT)
  target_compile_definitions(dr_mpi INTERFACE DR_FORMAT)
  target_link_libraries(dr_mpi INTERFACE fmt::fmt)
endif()
if (ENABLE_SYCL)
  target_link_libraries(dr_mpi INTERFACE oneDPL)
endif()
target_include_directories(dr_mpi INTERFACE . vendor)

# shp: sycl, single process, multi-gpu

add_library(dr_shp INTERFACE)
add_library(DR::shp ALIAS dr_shp)

target_include_directories(dr_shp INTERFACE . vendor)
target_compile_definitions(dr_shp INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
target_link_libraries(dr_shp INTERFACE range-v3 oneDPL fmt::fmt)

target_compile_options(dr_shp INTERFACE -fsycl)

# For use, see: https://github.com/illuhad/hipSYCL/blob/develop/doc/using-hipsycl.md#using-the-cmake-integration
# example: cmake .. -DhipSYCL_DIR=</hipsycl/install/lib/cmake/hipSYCL> -DHIPSYCL_TARGETS="<targets>"
if($(HIPSYCL_TARGETS))
  find_package(hipSYCL REQUIRED)
  add_sycl_to_target(TARGET dr_shp)
  target_compile_options(dr_shp INTERFACE --hipsycl-targets='cuda:sm_75')
endif()
