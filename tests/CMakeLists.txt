add_executable(reversability_test reversability_test.cpp)
target_link_libraries(reversability_test PUBLIC klft Kokkos::kokkos yaml-cpp)
