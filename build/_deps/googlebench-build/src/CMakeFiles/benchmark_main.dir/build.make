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
CMAKE_BINARY_DIR = /nfs/site/home/nowakmat/work/distributed-ranges-sort/build

# Include any dependencies generated for this target.
include _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/flags.make

_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.o: _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/flags.make
_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.o: _deps/googlebench-src/src/benchmark_main.cc
_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.o: _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.o"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.o -MF CMakeFiles/benchmark_main.dir/benchmark_main.cc.o.d -o CMakeFiles/benchmark_main.dir/benchmark_main.cc.o -c /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-src/src/benchmark_main.cc

_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_main.dir/benchmark_main.cc.i"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-src/src/benchmark_main.cc > CMakeFiles/benchmark_main.dir/benchmark_main.cc.i

_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_main.dir/benchmark_main.cc.s"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-src/src/benchmark_main.cc -o CMakeFiles/benchmark_main.dir/benchmark_main.cc.s

# Object files for target benchmark_main
benchmark_main_OBJECTS = \
"CMakeFiles/benchmark_main.dir/benchmark_main.cc.o"

# External object files for target benchmark_main
benchmark_main_EXTERNAL_OBJECTS =

_deps/googlebench-build/src/libbenchmark_main.a: _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/benchmark_main.cc.o
_deps/googlebench-build/src/libbenchmark_main.a: _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/build.make
_deps/googlebench-build/src/libbenchmark_main.a: _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libbenchmark_main.a"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_main.dir/cmake_clean_target.cmake
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/build: _deps/googlebench-build/src/libbenchmark_main.a
.PHONY : _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/build

_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/clean:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_main.dir/cmake_clean.cmake
.PHONY : _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/clean

_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/depend:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfs/site/home/nowakmat/work/distributed-ranges-sort /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-src/src /nfs/site/home/nowakmat/work/distributed-ranges-sort/build /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/googlebench-build/src/CMakeFiles/benchmark_main.dir/depend

