# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Applications/CLion 2019.1 EAP.app/Contents/bin/cmake/mac/bin/cmake"

# The command to remove a file.
RM = "/Applications/CLion 2019.1 EAP.app/Contents/bin/cmake/mac/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/paterno/repos/minigia

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/paterno/repos/minigia/cmake-build-release

# Include any dependencies generated for this target.
include CMakeFiles/independent_particle.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/independent_particle.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/independent_particle.dir/flags.make

CMakeFiles/independent_particle.dir/independent_particle_main.cc.o: CMakeFiles/independent_particle.dir/flags.make
CMakeFiles/independent_particle.dir/independent_particle_main.cc.o: ../independent_particle_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/paterno/repos/minigia/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/independent_particle.dir/independent_particle_main.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/independent_particle.dir/independent_particle_main.cc.o -c /Users/paterno/repos/minigia/independent_particle_main.cc

CMakeFiles/independent_particle.dir/independent_particle_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/independent_particle.dir/independent_particle_main.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/paterno/repos/minigia/independent_particle_main.cc > CMakeFiles/independent_particle.dir/independent_particle_main.cc.i

CMakeFiles/independent_particle.dir/independent_particle_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/independent_particle.dir/independent_particle_main.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/paterno/repos/minigia/independent_particle_main.cc -o CMakeFiles/independent_particle.dir/independent_particle_main.cc.s

CMakeFiles/independent_particle.dir/commxx.cc.o: CMakeFiles/independent_particle.dir/flags.make
CMakeFiles/independent_particle.dir/commxx.cc.o: ../commxx.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/paterno/repos/minigia/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/independent_particle.dir/commxx.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/independent_particle.dir/commxx.cc.o -c /Users/paterno/repos/minigia/commxx.cc

CMakeFiles/independent_particle.dir/commxx.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/independent_particle.dir/commxx.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/paterno/repos/minigia/commxx.cc > CMakeFiles/independent_particle.dir/commxx.cc.i

CMakeFiles/independent_particle.dir/commxx.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/independent_particle.dir/commxx.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/paterno/repos/minigia/commxx.cc -o CMakeFiles/independent_particle.dir/commxx.cc.s

# Object files for target independent_particle
independent_particle_OBJECTS = \
"CMakeFiles/independent_particle.dir/independent_particle_main.cc.o" \
"CMakeFiles/independent_particle.dir/commxx.cc.o"

# External object files for target independent_particle
independent_particle_EXTERNAL_OBJECTS =

independent_particle: CMakeFiles/independent_particle.dir/independent_particle_main.cc.o
independent_particle: CMakeFiles/independent_particle.dir/commxx.cc.o
independent_particle: CMakeFiles/independent_particle.dir/build.make
independent_particle: /usr/local/Cellar/open-mpi/4.0.0/lib/libmpi.dylib
independent_particle: /usr/local/lib/libomp.dylib
independent_particle: CMakeFiles/independent_particle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/paterno/repos/minigia/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable independent_particle"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/independent_particle.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/independent_particle.dir/build: independent_particle

.PHONY : CMakeFiles/independent_particle.dir/build

CMakeFiles/independent_particle.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/independent_particle.dir/cmake_clean.cmake
.PHONY : CMakeFiles/independent_particle.dir/clean

CMakeFiles/independent_particle.dir/depend:
	cd /Users/paterno/repos/minigia/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/paterno/repos/minigia /Users/paterno/repos/minigia /Users/paterno/repos/minigia/cmake-build-release /Users/paterno/repos/minigia/cmake-build-release /Users/paterno/repos/minigia/cmake-build-release/CMakeFiles/independent_particle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/independent_particle.dir/depend
