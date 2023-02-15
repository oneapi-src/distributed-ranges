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

  set(buildTypes Release Debug)

  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected (CMAKE_BUILD_TYPE), defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Release Debug ..." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${buildTypes})
  else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    if(NOT CMAKE_BUILD_TYPE IN_LIST buildTypes)
      message(WARNING "Unusual build type was set, please make sure it's a proper one. "
        "Only following are supported by default: ${buildTypes}.")
    endif()
  endif()

  set(CMAKE_C_FLAGS_DEBUG "-O0 -g -ggdb")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
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
  GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
  GIT_TAG 5300fe3
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
  GIT_TAG mdspan-0.5.0
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
  include_directories(test/gtest/include)

  include(GoogleTest)

  if(ENABLE_SYCL)
    add_subdirectory(examples/shp)
    add_subdirectory(test/gtest/shp)
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
    add_subdirectory(test/gtest/sycl-mpi)
  endif()

  add_subdirectory(examples/serial)
  add_subdirectory(test/gtest/serial)
  add_subdirectory(examples/mhp)
  add_subdirectory(test/gtest/cpu)
  add_subdirectory(test/gtest/mhp)

  # Requires clang, icpx/llvm nightly do not support the tools
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT ENABLE_SYCL)
    add_subdirectory(test/fuzz/cpu)
  endif()

endif()
