# CMake generated Testfile for 
# Source directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/mhp
# Build directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/benchmarks/gbench/mhp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[mhp-bench-1]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "-launcher=fork" "./mhp-bench" "--vector-size" "30000" "--rows" "100" "--columns" "100" "--check")
set_tests_properties([=[mhp-bench-1]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/mhp/CMakeLists.txt;30;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/mhp/CMakeLists.txt;0;")
add_test([=[mhp-bench-1-sycl]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "-launcher=fork" "./mhp-bench" "--vector-size" "30000" "--rows" "100" "--columns" "100" "--check" "--benchmark_filter=-.*DPL.*" "--sycl")
set_tests_properties([=[mhp-bench-1-sycl]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/mhp/CMakeLists.txt;34;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/benchmarks/gbench/mhp/CMakeLists.txt;0;")
