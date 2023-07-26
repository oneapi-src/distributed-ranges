# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-build"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix/tmp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix/src/range-v3-populate-stamp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix/src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix/src/range-v3-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix/src/range-v3-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/range-v3-subbuild/range-v3-populate-prefix/src/range-v3-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
