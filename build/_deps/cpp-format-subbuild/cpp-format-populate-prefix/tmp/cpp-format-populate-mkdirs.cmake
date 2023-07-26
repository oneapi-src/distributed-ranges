# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/tmp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
