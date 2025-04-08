cmake_minimum_required(VERSION 3.21.0)
project(klft VERSION 0.0.1 LANGUAGES CXX)

# Honor already set CMAKE_CXX_STANDARD.
# Otherwise default to C++17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

#
# Kokkos
#
# General Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(test_plaq_SU3 ${CMAKE_SOURCE_DIR}/tests/GaugeSU3/test_plaq_SU3.cpp)
target_link_libraries(test_plaq_SU3 Kokkos::kokkos)

add_executable(test_plaq_SU2 ${CMAKE_SOURCE_DIR}/tests/GaugeSU2/test_plaq_SU2.cpp)
target_link_libraries(test_plaq_SU2 Kokkos::kokkos)

add_executable(test_plaq_U1 ${CMAKE_SOURCE_DIR}/tests/GaugeU1/test_plaq_U1.cpp)
target_link_libraries(test_plaq_U1 Kokkos::kokkos)

add_executable(test_metropolis_SU3 ${CMAKE_SOURCE_DIR}/tests/GaugeSU3/test_metropolis_SU3.cpp)
target_link_libraries(test_metropolis_SU3 Kokkos::kokkos)

add_executable(test_metropolis_SU2 ${CMAKE_SOURCE_DIR}/tests/GaugeSU2/test_metropolis_SU2.cpp)
target_link_libraries(test_metropolis_SU2 Kokkos::kokkos)

add_executable(test_metropolis_U1 ${CMAKE_SOURCE_DIR}/tests/GaugeU1/test_metropolis_U1.cpp)
target_link_libraries(test_metropolis_U1 Kokkos::kokkos)
