# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-build"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix/tmp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix/src/dpl-populate-stamp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix/src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix/src/dpl-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix/src/dpl-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/dpl-subbuild/dpl-populate-prefix/src/dpl-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
