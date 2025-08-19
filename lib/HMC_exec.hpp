
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
  // hmcparams.print();

  // // template argument deduction and safety
  // static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  // static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank = HMCType::rank;
  constexpr static size_t Nc = HMCType::Nc;
  // static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
  //               (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  // constexpr const size_t Nd = rank;
  // // get the dimensions
  // const auto& dimensions = g_in.dimensions;
  // // first we check that all the parameters are correct
  // assert(hmcparams.Ndims == Nd);
  // assert(hmcparams.Nd == Nd);
  // assert(hmcparams.Nc == Nc);
  // assert(hmcparams.L0 == dimensions[0]);
  // assert(hmcparams.L1 == dimensions[1]);
  // if constexpr (Nd > 2) {
  //   assert(hmcparams.L2 == dimensions[2]);
  // }
  // if constexpr (Nd > 3) {
  //   assert(hmcparams.L3 == dimensions[3]);
  // }

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
    // measure the gauge observables
    // <<<<<<< HEAD
    //     measureGaugeObservables<rank, Nc>(hamiltonian_field.gauge_field,
    //                                       gaugeObsParams, step);
    //     addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    //     // TODO:make flushAllGaugeObservables append the Observables to the
    //     files ->
    //     // don't lose all progress when the simulation is interupted if (step
    //     % 50
    //     // == 0) {
    //     //   // flush every 50 steps as well to not lose data on program
    //     interuption
    //     //   // TODO: this should be set by the Params
    //     //   flushAllGaugeObservables(gaugeObsParams);
    //     // }
    //     DEBUG_LOG("Max unitarity defect: "
    //               <<
    //               unitarity_check<DGaugeFieldType>(hamiltonian_field.gauge_field)
    //               << "\n");
    //     if (step % 1000 == 0) {
    //       unitarity_restore<DGaugeFieldType>(hamiltonian_field.gauge_field);
    //     }
    //   }
    //   // flush the measurements to the files
    //   forceflushAllGaugeObservables(gaugeObsParams);
    //   forceflushSimulationLogs(simLogParams);
    //
    //   return 0;
    // }
    //
    // #define INSTANTIATE_HMC(R, N) \
//   template int \
//   run_HMC<DeviceGaugeFieldType<R, N>, DeviceAdjFieldType<R, N>, RNGType>(
    //   \
//       typename DeviceGaugeFieldType<R, N>::type, \
//       typename DeviceAdjFieldType<R, N>::type, const HMCParams &, \
//       GaugeObservableParams &, SimulationLoggingParams &, RNGType &);
    //
    // INSTANTIATE_HMC(4, 1);
    // INSTANTIATE_HMC(4, 2);
    // INSTANTIATE_HMC(3, 2);
    // INSTANTIATE_HMC(3, 1);
    // INSTANTIATE_HMC(2, 1);
    // INSTANTIATE_HMC(2, 2);
    //
    // // TODO: when SU3 is fully implemented, add Nc=3 here.
    //
    // } // namespace klft
    // =======
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

// template int run_HMC<DeviceGaugeFieldType<4, 1>, DeviceAdjFieldType<4, 1>>(
//     typename DeviceGaugeFieldType<4, 1>::type& g_in,
//     typename DeviceAdjFieldType<4, 1>::type& a_in, const HMCParams&
//     hmcparams, GaugeObservableParams& gaugeObsParams, const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<4, 2>, DeviceAdjFieldType<4, 2>>(
//     typename DeviceGaugeFieldType<4, 2>::type& g_in,
//     typename DeviceAdjFieldType<4, 2>::type& a_in, const HMCParams&
//     hmcparams, GaugeObservableParams& gaugeObsParams, const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<3, 1>, DeviceAdjFieldType<3, 1>>(
//     typename DeviceGaugeFieldType<3, 1>::type& g_in,
//     typename DeviceAdjFieldType<3, 1>::type& a_in, const HMCParams&
//     hmcparams, GaugeObservableParams& gaugeObsParams, const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<2, 1>, DeviceAdjFieldType<2, 1>>(
//     typename DeviceGaugeFieldType<2, 1>::type& g_in,
//     typename DeviceAdjFieldType<2, 1>::type& a_in, const HMCParams&
//     hmcparams, GaugeObservableParams& gaugeObsParams, const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<2, 2>, DeviceAdjFieldType<2, 2>>(
//     typename DeviceGaugeFieldType<2, 2>::type& g_in,
//     typename DeviceAdjFieldType<2, 2>::type& a_in, const HMCParams&
//     hmcparams, GaugeObservableParams& gaugeObsParams, const RNGType& rng);

// TODO: when SU3 is fully implemented, add Nc=3 here.

} // namespace klft
