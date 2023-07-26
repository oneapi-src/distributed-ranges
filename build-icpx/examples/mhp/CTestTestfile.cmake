# CMake generated Testfile for 
# Source directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp
# Build directory: /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/mhp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[mhp_dot_product]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "1" "./mhp_dot_product_benchmark" "-n" "1000")
set_tests_properties([=[mhp_dot_product]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;11;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[tile-example]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./tile-example")
set_tests_properties([=[tile-example]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;15;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[vector-add]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./vector-add")
set_tests_properties([=[vector-add]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;21;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[stencil-1d]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./stencil-1d")
set_tests_properties([=[stencil-1d]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;26;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;29;add_mhp_example;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[stencil-1d-array]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./stencil-1d-array")
set_tests_properties([=[stencil-1d-array]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;26;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;30;add_mhp_example;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[stencil-1d-pointer]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./stencil-1d-pointer")
set_tests_properties([=[stencil-1d-pointer]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;26;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;31;add_mhp_example;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[stencil-2d-matrix]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./stencil-2d-matrix")
set_tests_properties([=[stencil-2d-matrix]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;35;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[stencil-2d-matrix-rows]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./stencil-2d-matrix-rows")
set_tests_properties([=[stencil-2d-matrix-rows]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;39;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[wave_equation]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "4" "./wave_equation")
set_tests_properties([=[wave_equation]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;43;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
add_test([=[vector-add-ref]=] "/opt/intel/oneapi/mpi/2021.9.0/bin/mpiexec" "-n" "2" "./vector-add-ref")
set_tests_properties([=[vector-add-ref]=] PROPERTIES  _BACKTRACE_TRIPLES "/nfs/site/home/nowakmat/work/distributed-ranges-sort/CMakeLists.txt;150;add_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;49;add_mpi_test;/nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/CMakeLists.txt;0;")
