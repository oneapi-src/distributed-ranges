# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(cmake-applications)

find_package(MPI REQUIRED)
include(FetchContent)
FetchContent_Declare(
  dr
  GIT_REPOSITORY https://github.com/oneapi-src/distributed-ranges.git
  GIT_TAG main)
FetchContent_MakeAvailable(dr)

set(CMAKE_CXX_STANDARD 20)
add_executable(mhp-app mhp-app.cpp)
target_link_libraries(mhp-app MPI::MPI_CXX DR::mpi)
