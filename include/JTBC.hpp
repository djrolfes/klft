#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "SimulationLogging.hpp"
#include <queue>
#include <random>
#include <sstream>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct JTBCParams {
  // Define parameters for the JTBC algorithm
  index_t defect_length;             // size of the defect on the lattice
  GaugeMonomial_Params gauge_params; // HMC parameters´
  JTBCSimulationLoggingParams jtbcSimLogParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;

  std::string to_string() const {
    std::ostringstream oss;
    oss << "JTBCParams: "
        << "defect_length = " << defect_length;
    return oss.str();
  }
};

struct JTBCDecision {
  size_t long_term = 100;
  size_t short_term = 10;
  real_t sigma_weight = 4.0;
  real_t short_term_weight = 0.75;
  real_t t2E_target = 0.1; // TODO: make input yaml dependent
  real_t sp_max_deriv_max = 0.0;
  real_t sp_max_deriv_min = 0.0;

  std::queue<real_t> sp_max_diff_long;
  std::queue<real_t> sp_max_diff_short;

  real_t t2E;

  real_t average(std::queue<real_t> &q) {
    real_t sum = 0.0;
    std::queue<real_t> temp = q;
    size_t n = temp.size();
    while (!temp.empty()) {
      sum += temp.front();
      temp.pop();
    }
    return (n > 0) ? sum / static_cast<real_t>(n) : 0.0;
  }

  real_t stddev(std::queue<real_t> &q, real_t mean) {
    real_t sum = 0.0;
    std::queue<real_t> temp = q;
    size_t n = temp.size();
    while (!temp.empty()) {
      real_t diff = temp.front() - mean;
      sum += diff * diff;
      temp.pop();
    }
    return (n > 1) ? sqrt(sum / static_cast<real_t>(n - 1)) : 0.0;
  }

  real_t acceptance_probability() {
    if (t2E >= t2E_target) {
      return 0.0; // always reject if we are above the target
    }

    real_t sp_max_deriv_range = sp_max_deriv_max - sp_max_deriv_min;
    real_t avg_long = average(sp_max_diff_long);
    real_t avg_short = average(sp_max_diff_short);
    real_t std_long = stddev(sp_max_diff_long, avg_long);
    real_t pos_long = (avg_long - sp_max_deriv_min) / sp_max_deriv_range;
    real_t pos_short = (avg_short - sp_max_deriv_min) / sp_max_deriv_range;

    real_t long_term_factor = std::max(real_t(0.0), 2.0 * (pos_long - 0.5));
    real_t long_term_weight =
        Kokkos::exp(-sigma_weight * std_long / sp_max_deriv_range);
    real_t short_term_factor = 1.0 + short_term_weight * (2.0 * pos_short - 1);

    return std::max(real_t(0.0),
                    long_term_factor * long_term_weight * short_term_factor);
  }

  void push_data(WilsonFlowData wfdata) {
    push_sp_max_deriv(wfdata.sp_max_deriv);
    t2E = wfdata.t_sqrd_E;
  }

  void push_sp_max_deriv(real_t val) {
    sp_max_diff_long.push(val);
    sp_max_diff_short.push(val);
    if (sp_max_diff_long.size() > long_term) {
      sp_max_diff_long.pop();
    }
    if (sp_max_diff_short.size() > short_term) {
      sp_max_diff_short.pop();
    }
    sp_max_deriv_max = std::max(sp_max_deriv_max, val); // update max derivative
    sp_max_deriv_min = std::min(sp_max_deriv_min, val); // update min derivative
  }

  bool ready() {
    return (sp_max_diff_long.size() == long_term) &&
           (sp_max_diff_short.size() == short_term);
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
  real_t defect_value{1.0};
  JTBCDecision decision;

  JTBC() = delete; // default constructor is not allowed

  JTBC(JTBCParams &params_, HMCType &hmc_, RNG &rng_,
       std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : params(params_), rng(rng_), dist(dist_), mt(mt_), hmc(hmc_) {
    Kokkos::fence();
  }

  void measure(JTBCSimulationLoggingParams &jtbcSimLogParams, const size_t step,
               const int accept) {
    if (!is_PBC_step()) {
      // only log for JTBC steps
      addJTBCLogData(jtbcSimLogParams, step, defect_value, accept);
    }
    // measure the simulation logging observables
  }

  void measure(GaugeObservableParams &gaugeObsParams, size_t step) {
    // measure the gauge observables

    // if the defect value is close to 1, we measure the observables
    // for the JTBC case
    if (is_PBC_step()) {
      // PBC case
      measureGaugeObservables<DGaugeFieldType>(
          hmc.hamiltonian_field.gauge_field, gaugeObsParams, step);
    } else {
      // JTBC case
      if constexpr (Nd == 4) {
        WilsonFlowParams &wfparams = gaugeObsParams.wilson_flow_params;
        WilsonFlow<DGaugeFieldType> wf(hmc.hamiltonian_field.gauge_field,
                                       wfparams);
        wf.flow();
      }
    }
  }

  void decide_defect_value(WilsonFlowData wfdata) {
    // decide the defect value for the next HMC step
    decision.push_sp_max_deriv(wfdata.sp_max_deriv);
    if (!decision.ready()) {
      // not enough data to make a decision yet
      return;
    }

    real_t decide_defect_value = dist(mt);
    real_t acceptance_probability = decision.acceptance_probability();

    if (KLFT_VERBOSITY > 1) {
      Kokkos::printf("Decide defect value: %f, acceptance probability: %f\n",
                     decide_defect_value, acceptance_probability);
    }

    defect_value = 1.0;
    if (decide_defect_value < acceptance_probability) {
      // accept the new defect value
      defect_value = 0.0;
    }

    if (KLFT_VERBOSITY > 1) {
      printf("Decided defect value: %f\n", defect_value);
    }
    // set the defect value in the gauge field
    hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
        defect_value);
  }

  void measure(SimulationLoggingParams &simLogParams, index_t step,
               real_t acc_rate, bool accept, real_t time, real_t obs_time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time,
               obs_time);
  }

  bool is_PBC_step() { return (defect_value > 0.9999999); }

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
             JTBCSimulationLoggingParams &jtbcSimLogParams,
             SimulationLoggingParams &simLogParams) {

  Kokkos::Timer timer;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  for (size_t step = 0; step < int_params.nsteps; ++step) {
    timer.reset();

    if (!jtbc.is_PBC_step()) {
      --step;
    }

    int accept = jtbc.step();

    const real_t time = timer.seconds();
    timer.reset();
    // Gauge observables
    jtbc.measure(gaugeObsParams, step);
    const real_t obs_time = timer.seconds();
    jtbc.measure(jtbcSimLogParams, step, accept);
    jtbc.decide_defect_value(gaugeObsParams.wilson_flow_params.last_wfdata);

    flushAllGaugeObservables(gaugeObsParams, step, true);
    flushJTBCSimulationLogs(jtbcSimLogParams, step, true);

    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);
    jtbc.measure(simLogParams, step, acc_rate, accept, time, obs_time);
    Kokkos::printf("Step: %zu, accepted: %d, Acceptance rate: %f, Time: %f\n",
                   step, accept, acc_rate, time);
    flushSimulationLogs(simLogParams, step, true);
  }

  forceflushAllGaugeObservables(gaugeObsParams, true);
  forceflushJTBCSimulationLogs(jtbcSimLogParams, true);
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
