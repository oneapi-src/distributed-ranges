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
CMAKE_COMMAND = /opt/hpc_software/tools/cmake/3.26.0/bin/cmake

# The command to remove a file.
RM = /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /nfs/site/home/nowakmat/work/distributed-ranges-sort

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx

# Include any dependencies generated for this target.
include examples/shp/CMakeFiles/dot_product_benchmark.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/shp/CMakeFiles/dot_product_benchmark.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/shp/CMakeFiles/dot_product_benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include examples/shp/CMakeFiles/dot_product_benchmark.dir/flags.make

examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o: examples/shp/CMakeFiles/dot_product_benchmark.dir/flags.make
examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o: /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/shp/dot_product_benchmark.cpp
examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o: examples/shp/CMakeFiles/dot_product_benchmark.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o -MF CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o.d -o CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o -c /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/shp/dot_product_benchmark.cpp

examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.i"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/shp/dot_product_benchmark.cpp > CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.i

examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.s"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/shp/dot_product_benchmark.cpp -o CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.s

# Object files for target dot_product_benchmark
dot_product_benchmark_OBJECTS = \
"CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o"

# External object files for target dot_product_benchmark
dot_product_benchmark_EXTERNAL_OBJECTS =

examples/shp/dot_product_benchmark: examples/shp/CMakeFiles/dot_product_benchmark.dir/dot_product_benchmark.cpp.o
examples/shp/dot_product_benchmark: examples/shp/CMakeFiles/dot_product_benchmark.dir/build.make
examples/shp/dot_product_benchmark: /opt/intel/oneapi/tbb/2021.10.0/lib/intel64/gcc4.8/libtbb.so.12
examples/shp/dot_product_benchmark: _deps/cpp-format-build/libfmt.a
examples/shp/dot_product_benchmark: examples/shp/CMakeFiles/dot_product_benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dot_product_benchmark"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dot_product_benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/shp/CMakeFiles/dot_product_benchmark.dir/build: examples/shp/dot_product_benchmark
.PHONY : examples/shp/CMakeFiles/dot_product_benchmark.dir/build

examples/shp/CMakeFiles/dot_product_benchmark.dir/clean:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp && $(CMAKE_COMMAND) -P CMakeFiles/dot_product_benchmark.dir/cmake_clean.cmake
.PHONY : examples/shp/CMakeFiles/dot_product_benchmark.dir/clean

examples/shp/CMakeFiles/dot_product_benchmark.dir/depend:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfs/site/home/nowakmat/work/distributed-ranges-sort /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/shp /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp /nfs/site/home/nowakmat/work/distributed-ranges-sort/build-icpx/examples/shp/CMakeFiles/dot_product_benchmark.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/shp/CMakeFiles/dot_product_benchmark.dir/depend

