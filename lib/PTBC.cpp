#include "PTBC.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "InputParser.hpp"
#include "SimulationLogging.hpp"
#include <mpi.h>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {
int PTBC_execute(const std::string &input_file) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank %d of %d\n", rank, size);

  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, hmcParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, simLogParams)) {
    printf("Error parsing input file\n");
    return -1;
  }

  PTBC test = PTBC(PTBCParams(HMCParams(), 10, 5), rank);
  // Finalize Kokkos and MPI

  return 0;
}
} // namespace klft
