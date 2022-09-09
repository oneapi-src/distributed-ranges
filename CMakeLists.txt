cmake_minimum_required(VERSION 3.20)
project(
  distributed_ranges
  VERSION 0.1
  DESCRIPTION "Distributed ranges")

# Project wide defaults, not needed when another project uses the
# library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include(CheckLanguage)

  #
  # C++: generic configuration
  #
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD 17)

  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)

  # Debugging support
  #set(CMAKE_BUILD_TYPE "DEBUG")
  set(CMAKE_BUILD_TYPE "RELEASE")

  set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

  include(CTest)
endif()

add_subdirectory(include)

# Examples are not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include_directories(examples/include)

  add_subdirectory(examples/cpu)

endif()
