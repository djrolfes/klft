#include "GLOBAL.hpp"
#include <mpi.h>

int main(int argc, char **argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  std::string input_file;
  int ret = 1; // parse_args(argc, argv, input_file);
  if (ret < 0) {
    MPI_Finalize();
    return ret;
  }

  // Execute HMC
  int result = 2; // HMC_execute(input_file);

  // Finalize Kokkos and MPI
  Kokkos::finalize();
  MPI_Finalize();

  return result;
}
