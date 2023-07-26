# CMake generated Testfile for 
# Source directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/shp
# Build directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/benchmarks/gbench/shp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[shp-bench]=] "./shp-bench" "--vector-size" "20000")
set_tests_properties([=[shp-bench]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/shp/CMakeLists.txt;14;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/shp/CMakeLists.txt;0;")
