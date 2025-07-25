add_library(klft STATIC 
  Metropolis.cpp
  HMC_exec.cpp)
target_link_libraries(klft PUBLIC Kokkos::kokkos yaml-cpp)


add_executable(PTBC PTBC.cpp)
target_link_libraries(PTBC PUBLIC klft MPI::MPI_CXX Kokkos::kokkos yaml-cpp)
