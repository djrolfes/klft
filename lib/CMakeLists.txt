add_library(klft STATIC 
  Metropolis.cpp)
target_link_libraries(klft PUBLIC Kokkos::kokkos yaml-cpp)