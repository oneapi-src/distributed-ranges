# CMake generated Testfile for 
# Source directory: /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp
# Build directory: /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[shp-bench]=] "./shp-bench" "--vector-size" "20000")
set_tests_properties([=[shp-bench]=] PROPERTIES  _BACKTRACE_TRIPLES "/users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp/CMakeLists.txt;23;add_test;/users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp/CMakeLists.txt;0;")
