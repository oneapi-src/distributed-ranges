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
CMAKE_SOURCE_DIR = /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild

# Utility rule file for cpp-format-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/cpp-format-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cpp-format-populate.dir/progress.make

CMakeFiles/cpp-format-populate: CMakeFiles/cpp-format-populate-complete

CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-install
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-mkdir
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-download
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-patch
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-configure
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-build
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-install
CMakeFiles/cpp-format-populate-complete: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'cpp-format-populate'"
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E make_directory /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles/cpp-format-populate-complete
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-done

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update:
.PHONY : cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-build: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'cpp-format-populate'"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E echo_append
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-build

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-configure: cpp-format-populate-prefix/tmp/cpp-format-populate-cfgcmd.txt
cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-configure: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'cpp-format-populate'"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E echo_append
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-configure

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-download: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-gitinfo.txt
cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-download: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'cpp-format-populate'"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -P /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/tmp/cpp-format-populate-gitclone.cmake
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-download

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-install: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "No install step for 'cpp-format-populate'"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E echo_append
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-install

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'cpp-format-populate'"
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -Dcfgdir= -P /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/tmp/cpp-format-populate-mkdirs.cmake
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-mkdir

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-patch: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'cpp-format-populate'"
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E echo_append
	/opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-patch

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update:
.PHONY : cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-test: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No test step for 'cpp-format-populate'"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E echo_append
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-build && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -E touch /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-test

cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Performing update step for 'cpp-format-populate'"
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-src && /opt/hpc_software/tools/cmake/3.26.0/bin/cmake -P /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/cpp-format-populate-prefix/tmp/cpp-format-populate-gitupdate.cmake

cpp-format-populate: CMakeFiles/cpp-format-populate
cpp-format-populate: CMakeFiles/cpp-format-populate-complete
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-build
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-configure
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-download
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-install
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-mkdir
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-patch
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-test
cpp-format-populate: cpp-format-populate-prefix/src/cpp-format-populate-stamp/cpp-format-populate-update
cpp-format-populate: CMakeFiles/cpp-format-populate.dir/build.make
.PHONY : cpp-format-populate

# Rule to build all files generated by this target.
CMakeFiles/cpp-format-populate.dir/build: cpp-format-populate
.PHONY : CMakeFiles/cpp-format-populate.dir/build

CMakeFiles/cpp-format-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpp-format-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpp-format-populate.dir/clean

CMakeFiles/cpp-format-populate.dir/depend:
	cd /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild /nfs/site/home/nowakmat/work/distributed-ranges-sort/build/_deps/cpp-format-subbuild/CMakeFiles/cpp-format-populate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cpp-format-populate.dir/depend

