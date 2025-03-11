cmake_minimum_required(VERSION 3.15...3.29)
project(klft LANGUAGES CXX)
find_package(Kokkos REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Define options for each executable (default ON)
option(BUILD_METROPOLIS "Build the metropolis executable" ON)
option(BUILD_HMC        "Build the hmc executable" ON)
option(BUILD_PTBC_HMC   "Build the ptbc_hmc executable" ON)

# ----------------------------------------
# Group source files in lib/ based on file name patterns.
# ----------------------------------------

# All files in lib/
file(GLOB ALL_LIB_SOURCES ${CMAKE_SOURCE_DIR}/lib/*.cpp)

# Files that contain "PTBC" in their name
file(GLOB PTBC_SOURCES ${CMAKE_SOURCE_DIR}/lib/*PTBC*.cpp)

# Files that start with "HMC_" (i.e. HMC files) 
file(GLOB HMC_ALL_SOURCES ${CMAKE_SOURCE_DIR}/lib/HMC_*.cpp)
# Remove any PTBC files from the HMC group, so HMC_ALL_SOURCES now contains only non-PTBC HMC files.
list(REMOVE_ITEM HMC_ALL_SOURCES ${PTBC_SOURCES})

# Files that start with "Metropolis_"
file(GLOB METROPOLIS_SOURCES ${CMAKE_SOURCE_DIR}/lib/Metropolis_*.cpp)

# Now, create a list for the common library sources by removing these target‐specific files.
set(COMMON_SOURCES ${ALL_LIB_SOURCES})
list(REMOVE_ITEM COMMON_SOURCES ${PTBC_SOURCES} ${HMC_ALL_SOURCES} ${METROPOLIS_SOURCES})


# Build the common library target "klft"
add_library(klft ${COMMON_SOURCES})
target_compile_features(klft PRIVATE cxx_std_17)
target_link_libraries(klft Kokkos::kokkos m)

# Optionally build the Metropolis target
if(BUILD_METROPOLIS)
  add_library(metropolis_lib ${METROPOLIS_SOURCES})
  target_compile_features(metropolis_lib PRIVATE cxx_std_17)
  target_link_libraries(metropolis_lib klft Kokkos::kokkos m)
  
  add_executable(metropolis ${CMAKE_SOURCE_DIR}/metropolis.cpp)
  target_compile_features(metropolis PRIVATE cxx_std_17)
  target_link_libraries(metropolis metropolis_lib Kokkos::kokkos m)
endif()

# Optionally build the HMC target (non-PTBC)
if(BUILD_HMC)
  add_library(hmc_lib ${HMC_ALL_SOURCES})
  target_compile_features(hmc_lib PRIVATE cxx_std_17)
  target_link_libraries(hmc_lib klft Kokkos::kokkos m)
  
  add_executable(hmc ${CMAKE_SOURCE_DIR}/hmc.cpp)
  target_compile_features(hmc PRIVATE cxx_std_17)
  target_link_libraries(hmc hmc_lib Kokkos::kokkos m)
endif()

# Optionally build the PTBC_HMC target
if(BUILD_PTBC_HMC)
  add_library(ptbc_hmc_lib ${PTBC_SOURCES})
  target_compile_features(ptbc_hmc_lib PRIVATE cxx_std_17)
  target_link_libraries(ptbc_hmc_lib klft Kokkos::kokkos m)
  
  add_executable(ptbc_hmc ${CMAKE_SOURCE_DIR}/hmc_ptbc.cpp)
  target_compile_features(ptbc_hmc PRIVATE cxx_std_17)
  target_link_libraries(ptbc_hmc ptbc_hmc_lib Kokkos::kokkos m)
endif()
