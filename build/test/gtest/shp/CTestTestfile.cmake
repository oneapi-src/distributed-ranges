# CMake generated Testfile for 
# Source directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp
# Build directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/test/gtest/shp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[shp]=] "./shp-tests")
set_tests_properties([=[shp]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;33;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;37;add_shp_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;0;")
add_test([=[shp-3]=] "./shp-tests" "--devicesCount" "3")
set_tests_properties([=[shp-3]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;33;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;38;add_shp_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;0;")
add_test([=[shp-3-only]=] "./shp-tests-3" "--devicesCount" "3")
set_tests_properties([=[shp-3-only]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;33;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;39;add_shp_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/shp/CMakeLists.txt;0;")
