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
include examples/mhp/CMakeFiles/stencil-1d-pointer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/mhp/CMakeFiles/stencil-1d-pointer.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/mhp/CMakeFiles/stencil-1d-pointer.dir/progress.make

# Include the compile flags for this target's objects.
include examples/mhp/CMakeFiles/stencil-1d-pointer.dir/flags.make

examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o: examples/mhp/CMakeFiles/stencil-1d-pointer.dir/flags.make
examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o: /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/stencil-1d-pointer.cpp
examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o: examples/mhp/CMakeFiles/stencil-1d-pointer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o -MF CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o.d -o CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o -c /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/stencil-1d-pointer.cpp

examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.i"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/stencil-1d-pointer.cpp > CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.i

examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.s"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp && /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp/stencil-1d-pointer.cpp -o CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.s

# Object files for target stencil-1d-pointer
stencil__1d__pointer_OBJECTS = \
"CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o"

# External object files for target stencil-1d-pointer
stencil__1d__pointer_EXTERNAL_OBJECTS =

examples/mhp/stencil-1d-pointer: examples/mhp/CMakeFiles/stencil-1d-pointer.dir/stencil-1d-pointer.cpp.o
examples/mhp/stencil-1d-pointer: examples/mhp/CMakeFiles/stencil-1d-pointer.dir/build.make
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/mpi/2021.9.0/lib/libmpicxx.so
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/mpi/2021.9.0/lib/libmpifort.so
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/mpi/2021.9.0/lib/release/libmpi.so
examples/mhp/stencil-1d-pointer: /lib/x86_64-linux-gnu/libdl.a
examples/mhp/stencil-1d-pointer: /lib/x86_64-linux-gnu/librt.a
examples/mhp/stencil-1d-pointer: /lib/x86_64-linux-gnu/libpthread.a
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_intel_ilp64.so
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_intel_thread.so
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_core.so
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so
examples/mhp/stencil-1d-pointer: _deps/cpp-format-build/libfmt.a
examples/mhp/stencil-1d-pointer: /opt/intel/oneapi/tbb/2021.10.0/lib/intel64/gcc4.8/libtbb.so.12
examples/mhp/stencil-1d-pointer: examples/mhp/CMakeFiles/stencil-1d-pointer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stencil-1d-pointer"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stencil-1d-pointer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/mhp/CMakeFiles/stencil-1d-pointer.dir/build: examples/mhp/stencil-1d-pointer
.PHONY : examples/mhp/CMakeFiles/stencil-1d-pointer.dir/build

examples/mhp/CMakeFiles/stencil-1d-pointer.dir/clean:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp && $(CMAKE_COMMAND) -P CMakeFiles/stencil-1d-pointer.dir/cmake_clean.cmake
.PHONY : examples/mhp/CMakeFiles/stencil-1d-pointer.dir/clean

examples/mhp/CMakeFiles/stencil-1d-pointer.dir/depend:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfs/site/home/nowakmat/work/distributed-ranges-sort /nfs/site/home/nowakmat/work/distributed-ranges-sort/examples/mhp /nfs/site/home/nowakmat/work/distributed-ranges-sort/build /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/examples/mhp/CMakeFiles/stencil-1d-pointer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/mhp/CMakeFiles/stencil-1d-pointer.dir/depend

