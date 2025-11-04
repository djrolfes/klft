
//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/
#pragma once
#include <cstddef>

#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "SimulationLogging.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

template <typename HMCType>
int run_HMC(HMCType& hmc,
            const Integrator_Params& integratorParams,
            GaugeObservableParams& gaugeObsParams,
            SimulationLoggingParams& simLogParams) {
  // initiate and execute the HMC with the given parameters
  printf("Executing HMC ...");
  static_assert(isHMCClass<HMCType>::value,
                "HMCType must be a valid HMC class");
  // hmcparams.print();

  // // template argument deduction and safety
  // static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  // static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  // static_assert((rank ==
  // DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
  //               (Nc ==
  //               DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
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
      printf(
          "Step: %ld, accepted: %ld, Acceptance rate: %f, Time: "
          "%f\n",
          step, static_cast<size_t>(accept), acc_rate, time);
    }
    measureGaugeObservables<typename HMCType::DeviceGaugeFieldType>(
        hmc.hamiltonian_field.gauge_field, gaugeObsParams, step);
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    flushSimulationLogs(simLogParams, step, true);
    flushAllGaugeObservables(gaugeObsParams, step, true);
    flushIO<DeviceGaugeFieldType<HMCType::rank, HMCType::Nc>>(
        hmc.ioParams, step, hmc.hamiltonian_field.gauge_field);
    // flush the measurements to the files
    // if flush is set to 0, we flush with the  header at the end of
    // the simulation
  }

  forceflushSimulationLogs(simLogParams, true);
  forceflushAllGaugeObservables(gaugeObsParams, true);
  flushIO<DeviceGaugeFieldType<HMCType::rank, HMCType::Nc>>(
      hmc.ioParams, integratorParams.nsteps, hmc.hamiltonian_field.gauge_field,
      true);

  printf("Total Acceptance rate: %f, Accept %f Configs", acc_rate, acc_sum);
  return 0;
}

// template int run_HMC<DeviceGaugeFieldType<4, 1>,
// DeviceAdjFieldType<4, 1>>(
//     typename DeviceGaugeFieldType<4, 1>::type& g_in,
//     typename DeviceAdjFieldType<4, 1>::type& a_in, const
//     HMCParams& hmcparams, GaugeObservableParams& gaugeObsParams,
//     const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<4, 2>,
// DeviceAdjFieldType<4, 2>>(
//     typename DeviceGaugeFieldType<4, 2>::type& g_in,
//     typename DeviceAdjFieldType<4, 2>::type& a_in, const
//     HMCParams& hmcparams, GaugeObservableParams& gaugeObsParams,
//     const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<3, 1>,
// DeviceAdjFieldType<3, 1>>(
//     typename DeviceGaugeFieldType<3, 1>::type& g_in,
//     typename DeviceAdjFieldType<3, 1>::type& a_in, const
//     HMCParams& hmcparams, GaugeObservableParams& gaugeObsParams,
//     const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<2, 1>,
// DeviceAdjFieldType<2, 1>>(
//     typename DeviceGaugeFieldType<2, 1>::type& g_in,
//     typename DeviceAdjFieldType<2, 1>::type& a_in, const
//     HMCParams& hmcparams, GaugeObservableParams& gaugeObsParams,
//     const RNGType& rng);

// template int run_HMC<DeviceGaugeFieldType<2, 2>,
// DeviceAdjFieldType<2, 2>>(
//     typename DeviceGaugeFieldType<2, 2>::type& g_in,
//     typename DeviceAdjFieldType<2, 2>::type& a_in, const
//     HMCParams& hmcparams, GaugeObservableParams& gaugeObsParams,
//     const RNGType& rng);

// TODO: when SU3 is fully implemented, add Nc=3 here.

}  // namespace klft
