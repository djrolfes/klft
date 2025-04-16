add_executable(plaq plaq.cpp)
target_link_libraries(plaq PUBLIC Kokkos::kokkos)

add_executable(wloop_temporal wloop_temporal.cpp)
target_link_libraries(wloop_temporal PUBLIC Kokkos::kokkos)