# add_executable(reversability_test reversability_test.cpp)
# target_link_libraries(reversability_test PUBLIC klft Kokkos::kokkos yaml-cpp)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

# add_executable(wilsonflow_test wilsonflow_test.cpp)
# target_link_libraries(wilsonflow_test PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(wilsonflow_sp_test wilsonflow_sp_test.cpp)
target_link_libraries(wilsonflow_sp_test PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(gauge_inv_test_Topo gauge_inv_test_Topo.cpp)
target_link_libraries(gauge_inv_test_Topo PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(gauge_inv_test_action_density gauge_inv_test_action_density.cpp)
target_link_libraries(gauge_inv_test_action_density PUBLIC klft Kokkos::kokkos yaml-cpp)

# add_executable(cooling_test cooling_test.cpp)
# target_link_libraries(cooling_test PUBLIC klft Kokkos::kokkos yaml-cpp)

# build the Grouptests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Group ${CMAKE_CURRENT_BINARY_DIR}/Group)
