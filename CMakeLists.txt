# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(
  distributed_ranges
  VERSION 0.1
  DESCRIPTION "Distributed ranges")

include(FetchContent)

# Project wide defaults, not needed when another project uses the
# library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include(CheckLanguage)
  include(CTest)

  option(ENABLE_SYCL "Build sycl shp examples" OFF)
  option(ENABLE_SYCL_MPI "Build sycl mpi examples" OFF)
  option(ENABLE_CUDA "Build for cuda" OFF)
  option(ENABLE_FORMAT "Build with format library" ON)
  option(GCC_TOOLCHAIN, "GCC toolchain to be used by clang-based compilers" OFF)

  #
  # C++: generic configuration
  #
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD 20)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fconcepts-diagnostics-depth=10)
    add_compile_options(-Wno-stringop-overflow)
  endif()

  if(GCC_TOOLCHAIN)
    add_compile_options(--gcc-toolchain=${GCC_TOOLCHAIN})
    add_link_options(--gcc-toolchain=${GCC_TOOLCHAIN})
  endif()

  # Debugging support
  #set(CMAKE_BUILD_TYPE "Debug")
  set(CMAKE_BUILD_TYPE "Release")

  set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

  #
  # Common dependencies for examples
  #
  find_package(MKL REQUIRED)
  find_package(MPI REQUIRED)
  find_package(OpenMP)

  #
  # Google test
  #
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.0.0
    )
  FetchContent_MakeAvailable(cxxopts)

endif()

FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  GIT_TAG 0.12.0
)
FetchContent_MakeAvailable(range-v3)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(cpp-format)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG a18c489
)
FetchContent_MakeAvailable(mdspan)

function(add_mpi_test test_name name processes)
  add_test(NAME ${test_name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes} ${MPIEXEC_PREFLAGS} ./${name} ${ARGN} COMMAND_EXPAND_LISTS)
endfunction()

install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

add_subdirectory(include)

# Examples are not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include_directories(examples/include)
  include_directories(tests/include)

  include(GoogleTest)

  if(ENABLE_SYCL)
    add_subdirectory(examples/sycl)
    add_subdirectory(tests/sycl)
  endif()

  # shp has warnings, but do for everything else
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)

  # This must be appear after -Wall
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-Wno-tautological-constant-compare)
    # DPL uses deprecated API
    add_compile_options(-Wno-deprecated-declarations)
  endif()

  if (ENABLE_SYCL_MPI)
    add_subdirectory(tests/sycl-mpi)
  endif()

  add_subdirectory(examples/serial)
  add_subdirectory(tests/serial)
  add_subdirectory(examples/cpu)
  add_subdirectory(tests/cpu)

endif()
