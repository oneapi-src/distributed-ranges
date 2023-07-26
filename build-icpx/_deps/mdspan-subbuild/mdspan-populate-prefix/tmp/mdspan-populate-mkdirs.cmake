# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-build"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix/tmp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix/src"
  "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/_deps/mdspan-subbuild/mdspan-populate-prefix/src/mdspan-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
