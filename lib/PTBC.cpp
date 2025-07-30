#include "PTBC.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HamiltonianField.hpp"
#include "InputParser.hpp"
#include "SimulationLogging.hpp"
#include <mpi.h>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

// Helper macro to reduce repetition
#define INSTANTIATE_PREPARE_HFIELD(ND, NC)                                     \
  template HamiltonianField<                                                   \
      DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>,                      \
      DeviceAdjFieldType<ND, NC>>                                              \
  prepareHamiltonianField_PTBC<                                                \
      DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>,                      \
      DeviceAdjFieldType<ND, NC>, RNGType>(PTBCParams &, RNGType &);

// All combinations you care about
INSTANTIATE_PREPARE_HFIELD(2, 1)
INSTANTIATE_PREPARE_HFIELD(2, 2)
INSTANTIATE_PREPARE_HFIELD(3, 1)
INSTANTIATE_PREPARE_HFIELD(3, 2)
INSTANTIATE_PREPARE_HFIELD(4, 1)
INSTANTIATE_PREPARE_HFIELD(4, 2)

#undef INSTANTIATE_PREPARE_HFIELD

int PTBC_execute(const std::string &input_file) {
  index_t rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank %d of %d\n", rank, size);

  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  bool inputFilesCorrectly = (parseInputFile(input_file, gaugeObsParams) &&
                              parseInputFile(input_file, hmcParams) &&
                              parseInputFile(input_file, simLogParams) &&
                              parseInputFile(input_file, ptbcParams));
  if (!inputFilesCorrectly) {
    printf("Error parsing input file\n");
    return -1;
  }

  ptbcParams.hmc_params = hmcParams;
  size_t Nd = hmcParams.Ndims;
  size_t Nc = hmcParams.Nc;
  RNGType rng(hmcParams.seed + rank);
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);

  bool correct_NdNcCombination = false;
  using DGaugeFieldType = DeviceGaugeFieldType<2, 2, GaugeFieldKind::PTBC>;
  using DAdjFieldType = DeviceAdjFieldType<2, 2>;
#define FIELDTYPES(ND, NC)                                                     \
  if (Nd == ND && Nc == NC) {                                                  \
    using DGaugeFieldType =                                                    \
        DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>;                    \
    using DAdjFieldType = DeviceAdjFieldType<ND, NC>;                          \
    correct_NdNcCombination = true;                                            \
  }
  FIELDTYPES(2, 1)
  FIELDTYPES(2, 2)
  FIELDTYPES(3, 1)
  FIELDTYPES(3, 2)
  FIELDTYPES(4, 1)
  FIELDTYPES(4, 2)

  HamiltonianField<DGaugeFieldType, DAdjFieldType> hamiltonian_field =
      prepareHamiltonianField_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams,
                                                                   rng);
  PTBC<DGaugeFieldType, DAdjFieldType, RNGType> ptbc(
      ptbcParams, rank, hamiltonian_field, rng, dist, mt);

  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};
  // hmc loop
  for (size_t step = 0; step < hmcParams.nsteps; ++step) {
    timer.reset();

    // perform hmc_step
    accept = ptbc.step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n", step,
             static_cast<size_t>(accept), acc_rate, time);
    }
    // measure the gauge observables
    // ptbc.measure(gaugeObsParams, step);
    ptbc.measure(simLogParams, step, acc_rate, accept, time);
  }
  // flushAllGaugeObservables(gaugeObsParams);
  flushSimulationLogs(simLogParams);

  return 0;
}
} // namespace klft
