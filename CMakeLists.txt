# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(
  distributed_ranges
  VERSION 0.1
  DESCRIPTION "Distributed ranges")

include(FetchContent)

option(ENABLE_ISHMEM OFF)
# Project wide defaults, not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include(CheckLanguage)
  include(CTest)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24
    cmake_policy(SET CMP0135 NEW)
  endif()
  include(ExternalProject)

  option(DISABLED_TESTS "Run disabled tests" OFF)
  option(ENABLE_SYCL "Build with sycl support" OFF)
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
  endif()

  if(GCC_TOOLCHAIN)
    add_compile_options(--gcc-toolchain=${GCC_TOOLCHAIN})
    add_link_options(--gcc-toolchain=${GCC_TOOLCHAIN})
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(ENABLE_SYCL on)
  endif()

  if(ENABLE_SYCL)
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
    if(ENABLE_CUDA)
      add_compile_options(-fsycl-targets=nvptx64-nvidia-cuda
                          -Wno-error=unknown-cuda-version)
      add_link_options(-fsycl-targets=nvptx64-nvidia-cuda
                       -Wno-error=unknown-cuda-version)
    endif()
  endif()

  set(buildTypes Release Debug)

  if(NOT CMAKE_BUILD_TYPE)
    message(
      STATUS "No build type selected (CMAKE_BUILD_TYPE), defaulting to Release")
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "Choose the type of build, options are: Release Debug ..."
              FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${buildTypes})
  else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    if(NOT CMAKE_BUILD_TYPE IN_LIST buildTypes)
      message(
        WARNING
          "Unusual build type was set, please make sure it's a proper one. "
          "Only following are supported by default: ${buildTypes}.")
    endif()
  endif()

  set(CMAKE_C_FLAGS_DEBUG "-O0 -g -ggdb")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  # When we define NDEBUG, ranges-v3 uses printf, which is not ok in sycl
  # kernels. When we do not define NDEBUG, google bench warns about timing
  # error. Need to investigate if there is another way to stop ranges-v3 from
  # using printf
  #
  # set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG") set(CMAKE_CXX_FLAGS_RELEASE "-O3
  # -DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

  #
  # Common dependencies for examples
  #
  find_package(MKL REQUIRED)
  find_package(MPI REQUIRED)
  find_package(OpenMP)
  message("MPI exec set to ${MPIEXEC_EXECUTABLE}")

  #
  # Google test
  #
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1)
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  #
  # Command-line options parser
  #
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.0.0)
  FetchContent_MakeAvailable(cxxopts)

endif()

FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
  GIT_TAG 5300fe3)
FetchContent_MakeAvailable(range-v3)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 0b0f7cf)
FetchContent_MakeAvailable(cpp-format)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)

add_custom_target(mhp-all-tests)
add_custom_target(shp-all-tests)
add_custom_target(all-tests DEPENDS mhp-all-tests shp-all-tests)

function(add_mhp_ctest_impl)
  set(options OFFLOAD GDB SYCL DRLOGS TESTLABEL)
  set(oneValueArgs NAME TEST_NAME NPROC)
  set(multiValueArgs TARGS) # Test ARGumentS

  cmake_parse_arguments(AMC "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT DEFINED AMC_NAME)
    message(FATAL_ERROR "missing binary NAME")
  endif()

  set(nproc 1)
  if(DEFINED AMC_NPROC)
    set(nproc ${AMC_NPROC})
  endif()

  if(AMC_OFFLOAD OR ENABLE_ISHMEM)
    set(extra_mpiflags "-genv;I_MPI_OFFLOAD;1")
  endif()

  if(ENABLE_ISHMEM)
    set(wrapper_script ${CMAKE_BINARY_DIR}/bin/ishmrun)
    # set(extra_mpiflags "${extra_mpiflags};-genv;FI_PROVIDER;${OFI_PROVIDER}")
  endif()

  if(AMC_SYCL)
    set(sycl_param "--sycl")
  endif()

  # build a test name unless provided by user
  if(NOT DEFINED AMC_TEST_NAME)
    set(AMC_TEST_NAME "${AMC_NAME}")
    if(AMC_SYCL)
      set(AMC_TEST_NAME "${AMC_TEST_NAME}-sycl")
    endif()
    if(AMC_OFFLOAD)
      set(AMC_TEST_NAME "${AMC_TEST_NAME}-offload")
    endif()
    if(DEFINED AMC_NPROC) # add nproc to name only if not used default
      set(AMC_TEST_NAME "${AMC_TEST_NAME}-${nproc}")
    endif()
  endif()

  if(AMC_GDB)
    set(wrapper_script
        ${wrapper_script} gdb-oneapi -return-child-result --ex r -ex bt --ex q
        --batch --args)
    set(AMC_TEST_NAME "${AMC_TEST_NAME}-gdb")
    set(extra_mpiflags "${extra_mpiflags};-genv;ZET_ENABLE_PROGRAM_DEBUGGING;1")
  endif()

  if(AMC_DRLOGS)
    set(AMC_TEST_NAME "${AMC_TEST_NAME}-log")
    set(drlogs_param --log --logprefix=${AMC_TEST_NAME})
  endif()

  execute_process(
    COMMAND bash -c
            "${MPIEXEC_EXECUTABLE} --help 2>&1 | grep -q outfile-pattern"
    RESULT_VARIABLE DR_MPI_LACKS_FILEPATTERN_PARAM)

  if(NOT DR_MPI_LACKS_FILEPATTERN_PARAM)
    set(extra_mpiflags
        ${extra_mpiflags} -outfile-pattern "${AMC_TEST_NAME}.%r.out.log"
        -errfile-pattern "${AMC_TEST_NAME}.%r.err.log")
  endif()

  add_test(
    NAME ${AMC_TEST_NAME}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
      ${MPIEXEC_PREFLAGS} ${extra_mpiflags} ${wrapper_script} ./${AMC_NAME}
      ${drlogs_param} ${sycl_param} ${AMC_TARGS} COMMAND_EXPAND_LISTS)

  set_tests_properties(${AMC_TEST_NAME} PROPERTIES TIMEOUT 30)
  if(AMC_TESTLABEL)
    set_property(TEST ${AMC_TEST_NAME} PROPERTY LABELS TESTLABEL MHP)
  endif()
  add_dependencies(mhp-all-tests ${AMC_NAME})
endfunction()

function(add_mhp_auxiliary_ctests)
  add_mhp_ctest_impl(${ARGN} GDB)
  add_mhp_ctest_impl(${ARGN} GDB DRLOGS)
  add_mhp_ctest_impl(${ARGN} DRLOGS)
endfunction()

function(add_mhp_ctest)
  add_mhp_ctest_impl(${ARGN} TESTLABEL)
  add_mhp_auxiliary_ctests(${ARGN})
endfunction()

function(add_mhp_disabled_ctest)
  add_mhp_ctest_impl(${ARGN})
  add_mhp_auxiliary_ctests(${ARGN})
endfunction()

if(ENABLE_ISHMEM)

  set(OFI_PROVIDER
      "NONE"
      CACHE STRING "ofi prov: cxi verbs psm3")
  set_property(CACHE OFI_PROVIDER PROPERTY STRINGS cxi verbs psm3)

  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
  link_directories(${CMAKE_CURRENT_BINARY_DIR}/lib)

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_DEBUG_PARAM "--enable-debug")
  else()
    set(ENABLE_DEBUG_PARAM "")
  endif()

  # use it if there are no L0 headers installed
  if(ENABLE_L0)
    ExternalProject_Add(
      level-zero
      GIT_REPOSITORY https://github.com/oneapi-src/level-zero.git
      GIT_TAG v1.15.1
      INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
      CMAKE_CACHE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}"
    )

    set(SOS_L0_PARAM "-DL0_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}")
    set(SOS_DEPS "level-zero")
  else()
    set(SOS_L0_PARAM "")
    set(SOS_DEPS "")
  endif()

  if(ENABLE_OFI)
    ExternalProject_Add(
      ofi
      GIT_REPOSITORY git@github.com:ofiwg/libfabric.git
      GIT_TAG v1.18.0
      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/ofisrc
      DEPENDS ${SOS_DEPS}
      CONFIGURE_COMMAND
        cd ${CMAKE_CURRENT_BINARY_DIR}/ofisrc && ./autogen.sh && ./configure
        --prefix=${CMAKE_CURRENT_BINARY_DIR} ${ENABLE_DEBUG_PARAM}
        --with-ze=${CMAKE_CURRENT_BINARY_DIR} --with-dlopen=no
      BUILD_IN_SOURCE ON
      BUILD_COMMAND make
      INSTALL_COMMAND make install)
    set(OFI_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(SOS_DEPS ${SOS_DEPS} "ofi")
  endif()

  if(NOT DEFINED OFI_DIR)
    set(OFI_DIR /usr)
  endif()

  if(OFI_PROVIDER STREQUAL "cxi")
    set(SOS_OFI_PARAMS --enable-ofi-manual-progress --enable-mr-endpoint)
  elseif(OFI_PROVIDER STREQUAL "verbs" OR OFI_PROVIDER STREQUAL "psm3")
    set(SOS_OFI_PARAMS --enable-hard-polling)
  else()
    message(FATAL_ERROR "not supported OFI_PROVIDER:${OFI_PROVIDER}")
  endif()

  ExternalProject_Add(
    sos
    GIT_REPOSITORY git@github.com:Sandia-OpenSHMEM/SOS.git
    GIT_TAG main
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/sossrc
    DEPENDS ${SOS_DEPS}
    CONFIGURE_COMMAND
      cd ${CMAKE_CURRENT_BINARY_DIR}/sossrc && ./autogen.sh && ./configure
      --prefix=${CMAKE_CURRENT_BINARY_DIR} ${ENABLE_DEBUG_PARAM}
      --enable-pmi-simple --with-ofi=${OFI_DIR} --disable-fortran
      --enable-ofi-mr=basic --disable-ofi-inject --enable-ofi-hmem
      --disable-bounce-buffers ${SOS_OFI_PARAMS}
    BUILD_IN_SOURCE ON
    BUILD_COMMAND make
    INSTALL_COMMAND make install)

  ExternalProject_Add(
    ishmem
    GIT_REPOSITORY
      git@github.com:intel-innersource/libraries.runtimes.hpc.shmem.ishmem.git
    GIT_TAG main
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS sos
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=STRING:${CMAKE_BUILD_TYPE}
    CMAKE_CACHE_ARGS
      "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}"
      "-DSHMEM_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}" ${SOS_L0_PARAM})

  add_custom_target(shmem DEPENDS ishmem)

  function(target_link_ishmem exec_name)
    target_link_libraries(${exec_name} ze_loader pmi_simple sma)
    target_link_libraries(${exec_name} ${CMAKE_BINARY_DIR}/lib/libishmem.a)
    target_compile_definitions(${exec_name} PRIVATE DRISHMEM)
  endfunction()

endif()

function(add_shp_disabled_ctest test_name name)
  add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  add_dependencies(shp-all-tests ${name})
endfunction()

function(add_shp_ctest test_name name)
  add_shp_disabled_ctest(${test_name} ${name} ${ARGN})
  set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL SHP)
endfunction()

install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

add_subdirectory(include)

# Examples are not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include_directories(examples/include)
  include_directories(test/gtest/include)

  include(GoogleTest)

  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-error=stringop-overflow=)
  endif()

  # This must appear after -Wall
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Most code get warnings with -ffast-math
    add_compile_options(-Wno-tautological-constant-compare)
    # DPL turns on -fopenmp-simd, triggering optimization warnings in parallel
    # stl
    add_compile_options(-Wno-error=pass-failed)
    add_compile_options(-Wno-error=unknown-pragmas)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      add_compile_options(-qopenmp -qopt-streaming-stores=always
                          -qopt-zmm-usage=high)
      add_link_options(-qopenmp)
    endif()
  endif()

  add_subdirectory(test/gtest/mhp)

  if(ENABLE_SYCL)
    # disables rng::detail::box_compress::coalesce which causes rng::box to use
    # global non-const variable, which can not be used in SYCL kernels
    add_compile_definitions(RANGES_WORKAROUND_MSVC_249830)

    add_subdirectory(examples/shp)
    add_subdirectory(test/gtest/shp)
  endif()

  add_subdirectory(examples/serial)
  add_subdirectory(test/gtest/serial)
  add_subdirectory(examples/mhp)
  add_subdirectory(benchmarks/gbench)

  # Requires clang, icpx/llvm nightly do not support the tools
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT ENABLE_SYCL)
    add_subdirectory(test/fuzz/cpu)
  endif()

endif()
