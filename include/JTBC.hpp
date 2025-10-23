#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "SimulationLogging.hpp"
#include <random>
#include <sstream>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct JTBCParams {
  // Define parameters for the JTBC algorithm
  index_t defect_length;             // size of the defect on the lattice
  GaugeMonomial_Params gauge_params; // HMC parameters´
  // JTBCSimulationLoggingParams jtbcSimLogParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;

  std::string to_string() const {
    std::ostringstream oss;
    oss << "JTBCParams: "
        << ", defect_length = " << defect_length;
    return oss.str();
  }
};

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
class JTBC { // do I need the AdjFieldType here?
  // template argument deduction and safety
  static_assert(DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind ==
                GaugeFieldKind::JTBC);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((Nd == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  using HMCType = HMC<DGaugeFieldType, DAdjFieldType, RNG>;

public:
  JTBCParams &params; // parameters for the JTBC algorithm
  HMCType &hmc;
  RNG &rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;

  JTBC() = delete; // default constructor is not allowed

  JTBC(JTBCParams &params_, HMCType &hmc_, RNG &rng_,
       std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : params(params_), rng(rng_), dist(dist_), mt(mt_), hmc(hmc_) {
    Kokkos::fence();
  }

  // void measure(JTBCSimulationLoggingParams &jtbcSimLogParams,
  //              const size_t step) {
  //   // measure the simulation logging observables
  // }

  void measure(GaugeObservableParams &gaugeObsParams, size_t step) {
    // measure the gauge observables

    // if the defect value is close to 1, we measure the observables
    // for the JTBC case
    if (KLFT_VERBOSITY > 1) {
      printf("Measuring JTBC observables at step %zu\n", step);
    }
    measureGaugeObservables<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field,
                                             gaugeObsParams, step, 0, true);
  }

  void measure(SimulationLoggingParams &simLogParams, index_t step,
               real_t acc_rate, bool accept, real_t time, real_t obs_time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time,
               obs_time);
  }

  int step() {
    auto rtn = hmc.hmc_step();
    return rtn;
  }

  real_t getActionAroundDefect() {
    // calculate the plaquette around the defect
    // this is a placeholder function,
    // TODO: implement the actual action around
    // the defect

    auto action = WilsonAction<DGaugeFieldType>(
        hmc.hamiltonian_field.gauge_field, params.gauge_params.beta);

    return action;
  }
};

// below: Functions used to dispatch the JTBC algorithm

template <typename JTBCType>
int run_JTBC(JTBCType &jtbc, Integrator_Params &int_params,
             GaugeObservableParams &gaugeObsParams,
             // JTBCSimulationLoggingParams &jtbcSimLogParams,
             SimulationLoggingParams &simLogParams) {

  Kokkos::Timer timer;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  for (size_t step = 0; step < int_params.nsteps; ++step) {
    timer.reset();

    int accept = jtbc.step();
    MPI_Barrier(MPI_COMM_WORLD); // synchronize all ranks after step

    int jtbc_accept = jtbc.swap();

    const real_t time = timer.seconds();
    timer.reset();
    // Gauge observables
    jtbc.measure(gaugeObsParams, step);
    const real_t obs_time = timer.seconds();
    // jtbc.measure(jtbcSimLogParams, step);

    flushAllGaugeObservables(gaugeObsParams, step, true);
    // flushJTBCSimulationLogs(jtbcSimLogParams, step, true);

    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);
    jtbc.measure(simLogParams, step, acc_rate, accept, time, obs_time);
    Kokkos::printf("Step: %zu, accepted: %d, Acceptance rate: %f, Time: %f\n",
                   step, accept, acc_rate, time);
    flushSimulationLogs(simLogParams, step, true);
  }

  forceflushAllGaugeObservables(gaugeObsParams, true);
  // forceflushJTBCSimulationLogs(jtbcSimLogParams, true);
  forceflushSimulationLogs(simLogParams, true);

  return 0;
}

// #define INITIALIZE_JTBCPREPARE(ND, NC, RNG) \
//   template int run_JTBC<DeviceGaugeFieldType<ND, NC, GaugeFieldKind::JTBC>, \
//                         DeviceAdjFieldType<ND, NC>, RNG>( \
//       JTBCParams, RNG &, std::uniform_real_distribution<real_t>, \
//       std::mt19937);
// // INITIALIZE_JTBCPREPARE(2, 1, RNGType)
// // INITIALIZE_JTBCPREPARE(2, 2, RNGType)
// // // INITIALIZE_JTBCPREPARE(2, 3, RNGType)
// // INITIALIZE_JTBCPREPARE(3, 1, RNGType)
// // INITIALIZE_JTBCPREPARE(3, 2, RNGType)
// // INITIALIZE_JTBCPREPARE(3, 3, RNGType)
// INITIALIZE_JTBCPREPARE(4, 1, RNGType)
// INITIALIZE_JTBCPREPARE(4, 2, RNGType)
// // INITIALIZE_JTBCPREPARE(4, 3, RNGType)
// #undef INITIALIZE_JTBCPREPARE
//

} // namespace klft
