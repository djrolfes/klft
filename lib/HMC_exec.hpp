
#include <cstddef>

#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "SimulationLogging.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

template <typename HMCType>
int run_HMC(HMCType &hmc, const Integrator_Params &integratorParams,
            GaugeObservableParams &gaugeObsParams,
            SimulationLoggingParams &simLogParams) {
  // initiate and execute the HMC with the given parameters
  printf("Executing HMC ...");
  static_assert(isHMCClass<HMCType>::value,
                "HMCType must be a valid HMC class");

  // timer to measure the time per step
  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};
  // hmc loop
  for (size_t step = 0; step < integratorParams.nsteps; ++step) {
    timer.reset();

    // perform hmc_step
    accept = hmc.hmc_step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n", step,
             static_cast<size_t>(accept), acc_rate, time);
    }
    measureGaugeObservables<typename HMCType::DeviceGaugeFieldType>(
        hmc.hamiltonian_field.gauge_field, gaugeObsParams, step);
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    flushSimulationLogs(simLogParams, step, true);
    flushAllGaugeObservables(gaugeObsParams, step, true);
    // flush the measurements to the files
    // if flush is set to 0, we flush with the  header at the end of the
    // simulation
  }

  forceflushSimulationLogs(simLogParams, true);
  forceflushAllGaugeObservables(gaugeObsParams, true);

  printf("Total Acceptance rate: %f, Accept %f Configs", acc_rate, acc_sum);
  return 0;
}

} // namespace klft
