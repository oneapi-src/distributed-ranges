# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /users/suyashba/dist_ranges/distributed_ranges

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /users/suyashba/dist_ranges/distributed_ranges/build

# Include any dependencies generated for this target.
include benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/compiler_depend.make

# Include the progress variables for this target.
include benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/progress.make

# Include the compile flags for this target's objects.
include benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/flags.make

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/flags.make
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.o: /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp/shp-bench.cpp
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/users/suyashba/dist_ranges/distributed_ranges/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.o"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.o -MF CMakeFiles/shp-bench.dir/shp-bench.cpp.o.d -o CMakeFiles/shp-bench.dir/shp-bench.cpp.o -c /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp/shp-bench.cpp

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shp-bench.dir/shp-bench.cpp.i"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp/shp-bench.cpp > CMakeFiles/shp-bench.dir/shp-bench.cpp.i

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shp-bench.dir/shp-bench.cpp.s"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp/shp-bench.cpp -o CMakeFiles/shp-bench.dir/shp-bench.cpp.s

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/flags.make
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o: /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/distributed_vector.cpp
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/users/suyashba/dist_ranges/distributed_ranges/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o -MF CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o.d -o CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o -c /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/distributed_vector.cpp

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.i"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/distributed_vector.cpp > CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.i

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.s"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/distributed_vector.cpp -o CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.s

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/flags.make
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o: /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/dot_product.cpp
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/users/suyashba/dist_ranges/distributed_ranges/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o -MF CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o.d -o CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o -c /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/dot_product.cpp

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.i"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/dot_product.cpp > CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.i

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.s"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/dot_product.cpp -o CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.s

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/flags.make
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.o: /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/stream.cpp
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.o: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/users/suyashba/dist_ranges/distributed_ranges/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.o"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.o -MF CMakeFiles/shp-bench.dir/__/common/stream.cpp.o.d -o CMakeFiles/shp-bench.dir/__/common/stream.cpp.o -c /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/stream.cpp

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shp-bench.dir/__/common/stream.cpp.i"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/stream.cpp > CMakeFiles/shp-bench.dir/__/common/stream.cpp.i

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shp-bench.dir/__/common/stream.cpp.s"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && /opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/common/stream.cpp -o CMakeFiles/shp-bench.dir/__/common/stream.cpp.s

# Object files for target shp-bench
shp__bench_OBJECTS = \
"CMakeFiles/shp-bench.dir/shp-bench.cpp.o" \
"CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o" \
"CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o" \
"CMakeFiles/shp-bench.dir/__/common/stream.cpp.o"

# External object files for target shp-bench
shp__bench_EXTERNAL_OBJECTS =

benchmarks/gbench/shp/shp-bench: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/shp-bench.cpp.o
benchmarks/gbench/shp/shp-bench: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/distributed_vector.cpp.o
benchmarks/gbench/shp/shp-bench: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/dot_product.cpp.o
benchmarks/gbench/shp/shp-bench: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/__/common/stream.cpp.o
benchmarks/gbench/shp/shp-bench: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/build.make
benchmarks/gbench/shp/shp-bench: _deps/googlebench-build/src/libbenchmark.a
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mpi/2021.9.0/lib/libmpicxx.so
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mpi/2021.9.0/lib/libmpifort.so
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mpi/2021.9.0/lib/release/libmpi.so
benchmarks/gbench/shp/shp-bench: /lib/x86_64-linux-gnu/libdl.a
benchmarks/gbench/shp/shp-bench: /lib/x86_64-linux-gnu/librt.a
benchmarks/gbench/shp/shp-bench: /lib/x86_64-linux-gnu/libpthread.a
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mkl/2023.1.0/lib/intel64/libmkl_sycl.so
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mkl/2023.1.0/lib/intel64/libmkl_intel_ilp64.so
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mkl/2023.1.0/lib/intel64/libmkl_tbb_thread.so
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/mkl/2023.1.0/lib/intel64/libmkl_core.so
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8/libtbb.so.12
benchmarks/gbench/shp/shp-bench: _deps/cpp-format-build/libfmt.a
benchmarks/gbench/shp/shp-bench: /opt/intel/oneapi/tbb/2021.9.0/lib/intel64/gcc4.8/libtbb.so.12
benchmarks/gbench/shp/shp-bench: benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/users/suyashba/dist_ranges/distributed_ranges/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable shp-bench"
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/shp-bench.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/build: benchmarks/gbench/shp/shp-bench
.PHONY : benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/build

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/clean:
	cd /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp && $(CMAKE_COMMAND) -P CMakeFiles/shp-bench.dir/cmake_clean.cmake
.PHONY : benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/clean

benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/depend:
	cd /users/suyashba/dist_ranges/distributed_ranges/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /users/suyashba/dist_ranges/distributed_ranges /users/suyashba/dist_ranges/distributed_ranges/benchmarks/gbench/shp /users/suyashba/dist_ranges/distributed_ranges/build /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp /users/suyashba/dist_ranges/distributed_ranges/build/benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/gbench/shp/CMakeFiles/shp-bench.dir/depend

