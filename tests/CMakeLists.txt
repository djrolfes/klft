add_executable(reversability_test reversability_test.cpp)
target_link_libraries(reversability_test PUBLIC klft Kokkos::kokkos yaml-cpp)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

add_executable(wilsonflow_test wilsonflow_test.cpp)
target_link_libraries(wilsonflow_test PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(wilsonflow_sp_test wilsonflow_sp_test.cpp)
target_link_libraries(wilsonflow_sp_test PUBLIC klft Kokkos::kokkos yaml-cpp)

add_executable(cooling_test cooling_test.cpp)
target_link_libraries(cooling_test PUBLIC klft Kokkos::kokkos yaml-cpp)
add_executable(test_IO test_IO.cpp)
target_link_libraries(test_IO PUBLIC klft Kokkos::kokkos yaml-cpp)

# build the Grouptests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Group ${CMAKE_CURRENT_BINARY_DIR}/Group)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Fermions ${CMAKE_CURRENT_BINARY_DIR}/Fermions)