#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include "SimulationLogging.hpp"
#include <mpi.h>
#include <random>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct PTBCParams {
  // Define parameters for the PTBC algorithm
  index_t n_sims;
  std::vector<real_t> defects; // a vector that hold the different defect values
  index_t defect_size;         // size of the defect on the lattice
  HMCParams hmc_params;        // HMC parameters´
};

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
class PTBC { // do I need the AdjFieldType here?
  // template argument deduction and safety
  static_assert(DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind ==
                GaugeFieldKind::PTBC);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  using Update_Q = UpdatePositionGauge<rank, Nc, GaugeFieldKind::PTBC>;
  using Update_P = UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>;
  using HMCType = HMC<DGaugeFieldType, DAdjFieldType, RNG>;

public:
  PTBCParams params; // parameters for the PTBC algorithm
  HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field;
  HMCType hmc;
  std::shared_ptr<LeapFrog> integrator; // TODO: make integrator agnostic
  RNG rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;

  const index_t initial_index;
  index_t current_index;

  PTBC() = delete; // default constructor is not allowed

  PTBC(const PTBCParams &params_, const index_t initial_index_,
       HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field_,
       RNG rng_, std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : initial_index(initial_index_), current_index(initial_index_),
        params(params_), rng(rng_), dist(dist_), mt(mt_),
        hamiltonian_field(hamiltonian_field_),
        hmc(params.hmc_params, hamiltonian_field, integrator, rng, dist, mt) {
    device_id = Kokkos::device_id(); // default device id

    // host only fallback
    if (device_id == -1) {
      device_id = 0;
      partner_device_id = 0; // default to first two devices
    } else {
      partner_device_id =
          (device_id + 1) % Kokkos::num_devices(); // default partner device id
    }
    Update_Q update_q(hamiltonian_field.gauge_field,
                      hamiltonian_field.adjoint_field);
    Update_P update_p(hamiltonian_field.gauge_field,
                      hamiltonian_field.adjoint_field, params.hmc_params.beta);
    // the integrate might need to be passed into the run_HMC as an argument as
    // it contains a large amount of design decisions
    integrator =
        std::make_shared<LeapFrog>(params.hmc_params.nstepsGauge, true, nullptr,
                                   std::make_shared<Update_Q>(update_q),
                                   std::make_shared<Update_P>(update_p));
    init_hmc();
  }

  void init_hmc() {
    hmc.add_gauge_monomial(params.hmc_params.beta, 0);
    hmc.add_kinetic_monomial(0);
  }

  real_t getDefectValue() const {
    // return the defect value for the current index
    return params.defects[current_index];
  }

  void measure(GaugeObservableParams &gaugeObsParams, index_t step) {
    // measure the gauge observables
    measureGaugeObservables<rank, Nc>(hamiltonian_field.gauge_field,
                                      gaugeObsParams, step);
  }

  void measure(SimulationLoggingParams &simLogParams, index_t step,
               real_t acc_rate, bool accept, real_t time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
  }

  int step() { return hmc.hmc_step(); }

private:
  index_t device_id;
  index_t partner_device_id;
};

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
HamiltonianField<DGaugeFieldType, DAdjFieldType>
prepareHamiltonianField_PTBC(PTBCParams &ptbc_params, RNG &rng) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  static_assert(DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind ==
                GaugeFieldKind::PTBC);
  constexpr size_t Rank = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((Rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  static_assert(Nc > 0 && Nc < 3);
  static_assert(Rank >= 2 && Rank <= 4);

  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  HMCParams &hmc_params = ptbc_params.hmc_params;

  assert(hmc_params.Ndims == Rank);
  assert(hmc_params.Nc == Nc);
  defectParams<Rank> defect_params;
  defect_params.defect_length = ptbc_params.defect_size;
  bool coldStart = hmc_params.coldStart;

  SUN<Nc> sun_init = identitySUN<Nc>();
  if (coldStart) {
    if constexpr (Rank == 4) {
      return HField(typename DGaugeFieldType::type(
                        hmc_params.L0, hmc_params.L1, hmc_params.L2,
                        hmc_params.L3, identitySUN<Nc>(), defect_params),
                    typename DAdjFieldType::type(hmc_params.L0, hmc_params.L1,
                                                 hmc_params.L2, hmc_params.L3,
                                                 traceT(identitySUN<Nc>())));
    } else if constexpr (Rank == 3) {
      return HField(typename DGaugeFieldType::type(
                        hmc_params.L0, hmc_params.L1, hmc_params.L2,
                        identitySUN<Nc>(), defect_params),
                    typename DAdjFieldType::type(hmc_params.L0, hmc_params.L1,
                                                 hmc_params.L2,
                                                 traceT(identitySUN<Nc>())));
    } else if constexpr (Rank == 2) {
      return HField(typename DGaugeFieldType::type(hmc_params.L0, hmc_params.L1,
                                                   identitySUN<Nc>(),
                                                   defect_params),
                    typename DAdjFieldType::type(hmc_params.L0, hmc_params.L1,
                                                 traceT(identitySUN<Nc>())));
    } else {
      throw std::runtime_error("Unsupported Rank value");
    }
  } else {
    if constexpr (Rank == 4) {
      return HField(typename DGaugeFieldType::type(
                        hmc_params.L0, hmc_params.L1, hmc_params.L2,
                        hmc_params.L3, rng, hmc_params.rngDelta, defect_params),
                    typename DAdjFieldType::type(hmc_params.L0, hmc_params.L1,
                                                 hmc_params.L2, hmc_params.L3,
                                                 traceT(identitySUN<Nc>())));
    } else if constexpr (Rank == 3) {
      return HField(typename DGaugeFieldType::type(
                        hmc_params.L0, hmc_params.L1, hmc_params.L2, rng,
                        hmc_params.rngDelta, defect_params),
                    typename DAdjFieldType::type(hmc_params.L0, hmc_params.L1,
                                                 hmc_params.L2,
                                                 traceT(identitySUN<Nc>())));
    } else if constexpr (Rank == 2) {
      return HField(typename DGaugeFieldType::type(hmc_params.L0, hmc_params.L1,
                                                   rng, hmc_params.rngDelta,
                                                   defect_params),
                    typename DAdjFieldType::type(hmc_params.L0, hmc_params.L1,
                                                 traceT(identitySUN<Nc>())));
    } else {
      throw std::runtime_error("Unsupported Rank value");
    }
  }
}

} // namespace klft
