# mpi:  multi-process

add_library(dr_mpi INTERFACE)
add_library(DR::mpi ALIAS dr_mpi)

find_package(MPI REQUIRED)
find_package(MKL REQUIRED)
target_link_libraries(dr_mpi INTERFACE MPI::MPI_CXX fmt::fmt range-v3 std::mdspan MKL::MKL)
target_include_directories(dr_mpi INTERFACE . vendor)

# shp: sycl, single process, multi-gpu

add_library(dr_shp INTERFACE)
add_library(DR::shp ALIAS dr_shp)

FetchContent_Declare(
  dpl
  GIT_REPOSITORY https://github.com/oneapi-src/oneDPL.git
  GIT_TAG oneDPL-2021.7.0-release
  )
FetchContent_MakeAvailable(dpl)

target_include_directories(dr_shp INTERFACE dr)
target_compile_definitions(dr_shp INTERFACE _GLIBCXX_USE_TBB_PAR_BACKEND=0)
target_link_libraries(dr_shp INTERFACE oneDPL)

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  target_compile_options(dr_shp INTERFACE -fsycl -Wno-deprecated-declarations)
endif()

# For use, see: https://github.com/illuhad/hipSYCL/blob/develop/doc/using-hipsycl.md#using-the-cmake-integration
# example: cmake .. -DhipSYCL_DIR=</hipsycl/install/lib/cmake/hipSYCL> -DHIPSYCL_TARGETS="<targets>"
if($(HIPSYCL_TARGETS))
  find_package(hipSYCL REQUIRED)
  add_sycl_to_target(TARGET dr_shp)
  target_compile_options(dr_shp INTERFACE --hipsycl-targets='cuda:sm_75')
endif()
