cmake_minimum_required(VERSION 3.20)
project(
  distributed_ranges
  VERSION 0.1
  DESCRIPTION "Distributed ranges")

# Project wide defaults, not needed when another project uses the
# library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include(CheckLanguage)
  include(CTest)
  include(FetchContent)

  #
  # C++: generic configuration
  #
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD 20)

  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-Wno-tautological-constant-compare)
  endif()

  # Debugging support
  #set(CMAKE_BUILD_TYPE "DEBUG")
  set(CMAKE_BUILD_TYPE "RELEASE")

  set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

  #
  # Common dependencies for examples
  #
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

endif()

function(add_mpi_test test_name name processes)
  add_test(NAME ${test_name} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes} ${MPIEXEC_PREFLAGS} ./${name} ${ARGN} COMMAND_EXPAND_LISTS)
endfunction()

add_subdirectory(include)

# Examples are not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include_directories(examples/include)

  add_subdirectory(examples/serial)
  add_subdirectory(examples/cpu)

  add_subdirectory(tests)

endif()
