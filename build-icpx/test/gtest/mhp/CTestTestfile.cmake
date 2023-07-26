# CMake generated Testfile for 
# Source directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp
# Build directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/test/gtest/mhp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[mhp-quick-test-1]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "./mhp-quick-test")
set_tests_properties([=[mhp-quick-test-1]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;35;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-quick-test-2]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./mhp-quick-test")
set_tests_properties([=[mhp-quick-test-2]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;36;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-quick-test-1]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "./mhp-quick-test" "--sycl")
set_tests_properties([=[mhp-sycl-quick-test-1]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;37;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-quick-test-2]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./mhp-quick-test" "--sycl")
set_tests_properties([=[mhp-sycl-quick-test-2]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;38;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-tests-1]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "./mhp-tests")
set_tests_properties([=[mhp-tests-1]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;44;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-tests-2]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./mhp-tests")
set_tests_properties([=[mhp-tests-2]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;47;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-tests-3]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "3" "./mhp-tests")
set_tests_properties([=[mhp-tests-3]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;48;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-tests-4]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "4" "./mhp-tests")
set_tests_properties([=[mhp-tests-4]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;49;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-tests-3-only]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "3" "./mhp-tests-3")
set_tests_properties([=[mhp-tests-3-only]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;50;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-tests-1]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "./mhp-tests" "--sycl" "--gtest_filter=-*Slide*")
set_tests_properties([=[mhp-sycl-tests-1]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;55;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-tests-2]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./mhp-tests" "--sycl" "--gtest_filter=-*Slide*")
set_tests_properties([=[mhp-sycl-tests-2]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;59;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-tests-3]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "3" "./mhp-tests" "--sycl" "--gtest_filter=-*Slide*")
set_tests_properties([=[mhp-sycl-tests-3]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;60;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-tests-4]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "4" "./mhp-tests" "--sycl" "--gtest_filter=-*Slide*")
set_tests_properties([=[mhp-sycl-tests-4]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;61;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
add_test([=[mhp-sycl-tests-3-only]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "3" "./mhp-tests-3" "--sycl" "--gtest_filter=-*Slide*")
set_tests_properties([=[mhp-sycl-tests-3-only]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;62;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/test/gtest/mhp/CMakeLists.txt;0;")
