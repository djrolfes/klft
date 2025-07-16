add_executable(reversability_test reversability_test.cpp)
target_link_libraries(reversability_test PUBLIC klft Kokkos::kokkos yaml-cpp)
#
# build the Grouptests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Group ${CMAKE_CURRENT_BINARY_DIR}/Group)
