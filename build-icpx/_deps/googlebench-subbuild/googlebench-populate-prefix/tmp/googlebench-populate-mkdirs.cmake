# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-build"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix/tmp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix/src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
