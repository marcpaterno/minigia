cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(MINIGIA LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno  -march=native -mtune=native")
list(APPEND CMAKE_MODULE_PATH "${MINIGIA_SOURCE_DIR}/CMake")

if(DEFINES)
  add_definitions(${DEFINES})
endif()

find_package(MPI REQUIRED CXX)
find_package(Boost 1.66 REQUIRED COMPONENTS system)
find_package(OpenMP REQUIRED CXX)
find_package(FFTW3 REQUIRED)

include_directories(${FFTW3_INCLUDE_DIR})
set(PARALLEL_FFTW_LIBRARIES ${FFTW3_MPI_LIBRARIES} ${FFTW3_LIBRARIES})
configure_file("${MINIGIA_SOURCE_DIR}/bunch_data_paths.h.in"
               "${MINIGIA_BINARY_DIR}/bunch_data_paths.h" IMMEDIATE)

add_executable(independent_particle
               independent_particle_main.cc
               commxx.cc)
target_include_directories(independent_particle PRIVATE ${MINIGIA_BINARY_DIR})
target_include_directories(independent_particle PRIVATE vectorclass)
target_include_directories(independent_particle PRIVATE eigen)
target_link_libraries(independent_particle PRIVATE MPI::MPI_CXX)
target_link_libraries(independent_particle PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(independent_particle PRIVATE boost_system)

find_library(BENCHMARK benchmark)
add_executable(independent_particle_benchmark
               independent_particle_benchmark.cc
               commxx.cc)
target_include_directories(independent_particle_benchmark PRIVATE ${MINIGIA_BINARY_DIR} vectorclass eigen)
target_link_libraries(independent_particle_benchmark PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX boost_system ${BENCHMARK})

add_executable(populate
               populate.cc
               commxx.cc)
target_include_directories(populate PRIVATE ${MINIGIA_BINARY_DIR})
target_include_directories(populate PRIVATE eigen)
target_link_libraries(populate PRIVATE MPI::MPI_CXX)
target_link_libraries(populate PRIVATE OpenMP::OpenMP_CXX)

add_subdirectory(fmt)
#add_subdirectory(fftwpp)

#add_executable(fft
#               fft-main.cc
#               commxx.cc
#               distributed_fft3d.cc)
#target_include_directories(fft PRIVATE ${FFTW3_INCLUDE_DIR})
#target_include_directories(fft PRIVATE eigen)
#target_link_libraries(fft PRIVATE ${PARALLEL_FFTW_LIBRARIES})
#target_link_libraries(fft PRIVATE MPI::MPI_CXX)
#target_link_libraries(fft PRIVATE fftwpp)
#target_link_libraries(fft PRIVATE fmt)

#add_executable(space_charge_3d
#               space_charge_3d.cc
#               space_charge_3d_open_hockney.cc
#               space_charge_3d_open_hockney_eigen.cc
#               commxx.cc
#               commxx_divider.cc
#               core_diagnostics.cc
#               deposit.cc
#               distributed_fft3d.cc
#               distributed_fft3d_eigen.cc
#               distributed_rectangular_grid.cc
#               #distributed_rectangular_grid_eigen.cc
#               rectangular_grid.cc
#               rectangular_grid_domain.cc
#               #rectangular_grid_domain_eigen.cc
#               space_charge_3d.cc
#               space_charge_3d_open_hockney.cc
#               )
#target_include_directories(space_charge_3d PRIVATE ${MINIGIA_BINARY_DIR})
#target_include_directories(space_charge_3d PRIVATE vectorclass)
#target_include_directories(space_charge_3d PRIVATE eigen)
#target_link_libraries(space_charge_3d PRIVATE MPI::MPI_CXX)
#target_link_libraries(space_charge_3d PRIVATE ${PARALLEL_FFTW_LIBRARIES})
