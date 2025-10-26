add_library(klft STATIC 
  Metropolis.cpp
  HMC_exec.cpp
  PTBC.cpp
  JTBC.cpp)
target_link_libraries(klft PUBLIC MPI::MPI_CXX Kokkos::kokkos yaml-cpp)
