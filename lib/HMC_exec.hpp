
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeFieldHelper.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include "SimulationLogging.hpp"
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"
#include <cstddef>
#include <memory>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
int run_HMC(typename DGaugeFieldType::type g_in,
            typename DAdjFieldType::type a_in, const HMCParams &hmcparams,
            GaugeObservableParams &gaugeObsParams,
            SimulationLoggingParams &simLogParams, RNG &rng) {
  // initiate and execute the HMC with the given parameters
  printf("Executing HMC ...");
  hmcparams.print();

  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  constexpr const size_t Nd = rank;
  // get the dimensions

  // define fields and update routines
  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  using Update_Q = UpdatePositionGauge<Nd, Nc>;
  using Update_P = UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>;

  HField hamiltonian_field = HField(g_in, a_in);
  // after the move, the gauge and adjoint fields are no longer valid
  const auto &dimensions = hamiltonian_field.gauge_field.dimensions;
  // first we check that all the parameters are correct
  assert(hmcparams.Ndims == Nd);
  assert(hmcparams.Nd == Nd);
  assert(hmcparams.Nc == Nc);
  assert(hmcparams.L0 == dimensions[0]);
  assert(hmcparams.L1 == dimensions[1]);
  if constexpr (Nd > 2) {
    assert(hmcparams.L2 == dimensions[2]);
  }
  if constexpr (Nd > 3) {
    assert(hmcparams.L3 == dimensions[3]);
  }

  Update_Q update_q(hamiltonian_field.gauge_field,
                    hamiltonian_field.adjoint_field);
  Update_P update_p(hamiltonian_field.gauge_field,
                    hamiltonian_field.adjoint_field, hmcparams.beta);
  // the integrate might need to be passed into the run_HMC as an argument as it
  // contains a large amount of design decisions
  std::shared_ptr<LeapFrog> leap_frog =
      std::make_shared<LeapFrog>(hmcparams.nstepsGauge, true, nullptr,
                                 std::make_shared<Update_Q>(update_q),
                                 std::make_shared<Update_P>(update_p));

  // now define and run the hmc
  std::mt19937 mt(hmcparams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);
  using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNG>;
  HMC hmc(hmcparams, hamiltonian_field, leap_frog, rng, dist, mt);
  hmc.add_gauge_monomial(hmcparams.beta, 0);
  hmc.add_kinetic_monomial(0);

  // timer to measure the time per step
  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};
  // hmc loop
  for (size_t step = 0; step < hmcparams.nsteps; ++step) {
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
    measureGaugeObservables<rank, Nc>(hamiltonian_field.gauge_field,
                                      gaugeObsParams, step);
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    // TODO:make flushAllGaugeObservables append the Observables to the files ->
    // don't lose all progress when the simulation is interupted if (step % 50
    // == 0) {
    //   // flush every 50 steps as well to not lose data on program interuption
    //   // TODO: this should be set by the Params
    //   flushAllGaugeObservables(gaugeObsParams);
    // }
    DEBUG_LOG("Max unitarity defect: "
              << unitarity_check<DGaugeFieldType>(hamiltonian_field.gauge_field)
              << "\n");
    if (step % 1000 == 0) {
      unitarity_restore<DGaugeFieldType>(hamiltonian_field.gauge_field);
    }
  }
  // flush the measurements to the files
  flushAllGaugeObservables(gaugeObsParams);
  flushSimulationLogs(simLogParams);

  return 0;
}

#define INSTANTIATE_HMC(R, N)                                                  \
  template int                                                                 \
  run_HMC<DeviceGaugeFieldType<R, N>, DeviceAdjFieldType<R, N>, RNGType>(      \
      typename DeviceGaugeFieldType<R, N>::type,                               \
      typename DeviceAdjFieldType<R, N>::type, const HMCParams &,              \
      GaugeObservableParams &, SimulationLoggingParams &, RNGType &);

INSTANTIATE_HMC(4, 1);
INSTANTIATE_HMC(4, 2);
INSTANTIATE_HMC(3, 2);
INSTANTIATE_HMC(3, 1);
INSTANTIATE_HMC(2, 1);
INSTANTIATE_HMC(2, 2);

// TODO: when SU3 is fully implemented, add Nc=3 here.

} // namespace klft
