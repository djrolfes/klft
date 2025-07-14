#include "AdjointSUN.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeFieldHelper.hpp"
#include "HMC.hpp"
#include "HamiltonianField.hpp"
#include "InputParser.hpp"
#include "Integrator.hpp"
#include "SimulationLogging.hpp"
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"
#include <cassert>
#include <cstddef>
#include <getopt.h>
#include <memory>

using namespace klft;

#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

#define HLINE                                                                  \
  "====================================================================\n"

int parse_args(int argc, char **argv, std::string &input_file) {
  // Defaults
  input_file = "input.yaml";

  const std::string help_string =
      "  -f <file_name> --filename <file_name>\n"
      "     Name of the input file.\n"
      "     Default: input.yaml\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"filename", required_argument, NULL, 'f'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "f:h", long_options, &option_index)) !=
         -1)
    switch (c) {
    case 'f':
      input_file = optarg;
      break;
    case 'h':
      printf("%s", help_string.c_str());
      return -2;
      break;
    case 0:
      break;
    default:
      printf("%s", help_string.c_str());
      return -1;
      break;
    }
  return 0;
}

// for now, the test only allows for Nd=4
int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("HMC for SU(N) gauge fields\n");
  printf(HLINE);

  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);

  Kokkos::initialize(argc, argv);
  int rc;
  std::string input_file;
  rc = parse_args(argc, argv, input_file);
  if (rc == 2) {
    Kokkos::finalize();
    return 0;
  }
  if (rc != 0) {
    Kokkos::finalize();
    return rc;
  }

  HMCParams hmcParams;
  SimulationLoggingParams simLogParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, hmcParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, simLogParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }

  constexpr const index_t Nd = 4;
  const constexpr index_t Nc = 2;
  assert(hmcParams.Ndims == 4 && hmcParams.Nc == 2);

  if (Nd == 4 && Nc == 2) {

    RNGType rng(hmcParams.seed);
    using DGaugeFieldType = DeviceGaugeFieldType<Nd, Nc>;
    using DAdjFieldType = DeviceAdjFieldType<Nd, Nc>;

    typename DGaugeFieldType::type dev_g_SU2_4D(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, rng,
                                                hmcParams.rngDelta);
    typename DAdjFieldType::type dev_a_SU2_4D(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              traceT(identitySUN<Nc>()));
    using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
    using Update_Q = UpdatePositionGauge<Nd, Nc>;
    using Update_P = UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>;

    auto gauge_ptr = std::move(
        std::make_unique<typename DGaugeFieldType::type>(dev_g_SU2_4D));
    auto adjoint_ptr =
        std::move(std::make_unique<typename DAdjFieldType::type>(dev_a_SU2_4D));
    HField hamiltonian_field =
        HField(std::move(gauge_ptr), std::move(adjoint_ptr));
    // after the move, the gauge and adjoint fields are no longer valid
    const auto &dimensions = hamiltonian_field.gauge_field().dimensions;

    Update_Q update_q(hamiltonian_field.gauge_field(),
                      hamiltonian_field.adjoint_field());
    Update_P update_p(hamiltonian_field.gauge_field(),
                      hamiltonian_field.adjoint_field(), hmcParams.beta);
    // the integrate might need to be passed into the run_HMC as an argument as
    // it contains a large amount of design decisions
    std::shared_ptr<LeapFrog> leap_frog =
        std::make_shared<LeapFrog>(hmcParams.nstepsGauge, true, nullptr,
                                   std::make_shared<Update_Q>(update_q),
                                   std::make_shared<Update_P>(update_p));

    // now define and run the hmc
    std::mt19937 mt(hmcParams.seed);
    std::uniform_real_distribution<real_t> dist(0.0, 1.0);
    using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
    HMC hmc(hmcParams, hamiltonian_field, leap_frog, rng, dist, mt);
    hmc.add_gauge_monomial(hmcParams.beta, 0);
    hmc.add_kinetic_monomial(0);

    // timer to measure the time per step
    Kokkos::Timer timer;
    bool accept;
    real_t acc_sum{0.0};
    real_t acc_rate{0.0};

    for (size_t step = 0; step < hmcParams.nsteps; ++step) {

      printf("tau: %f\n", hmcParams.tau);
      timer.reset();
      // perform hmc_step
      accept = hmc.hmc_step();
      const real_t time = timer.seconds();
      acc_sum += static_cast<real_t>(accept);
      acc_rate = acc_sum / static_cast<real_t>(step + 1);
      if (KLFT_VERBOSITY > 0) {
        printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n",
               step, static_cast<size_t>(accept), acc_rate, time);
      }
      measureGaugeObservables<Nd, Nc>(hamiltonian_field.gauge_field(),
                                      gaugeObsParams, step);
      addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);

      // printf("Reverse step with tau = -1\n");
      // hmcParams.tau *= -1;
      // printf("tau: %f\n", hmcParams.tau);
      //
      // timer.reset();
      // // perform hmc_step
      // accept = hmc.hmc_step(true);
      // const real_t time1 = timer.seconds();
      // acc_sum += static_cast<real_t>(accept);
      // acc_rate = acc_sum / static_cast<real_t>(step + 1);
      // if (KLFT_VERBOSITY > 0) {
      //   printf(
      //       "Reverse Step: %ld, accepted: %ld, Acceptance rate: %f, Time:
      //       %f\n", step, static_cast<size_t>(accept), acc_rate, time);
      // }
      // addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
      // printf("resetting gauge to before reverse step\n");
      // hmc.reset_gauge_field();
      //
      // hmcParams.tau *= -1;
    }

    flushAllGaugeObservables(gaugeObsParams);
    flushSimulationLogs(simLogParams);
  }
  Kokkos::finalize();

  return rc;
}
