#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include "SimulationLogging.hpp"
#include <memory>
#include <mpi.h>
#include <random>
#include <variant>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct PTBCParams {
  // Define parameters for the PTBC algorithm
  index_t n_sims;
  std::vector<real_t> defects; // a vector that hold the different defect values
  index_t defect_size;         // size of the defect on the lattice
  HMCParams hmc_params;        // HMC parameters´
  real_t defect_value;         // value of the defect
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
  // std::shared_ptr<LeapFrog> integrator; // TODO: make integrator agnostic
  RNG &rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;

  const index_t initial_index;
  index_t current_index;

  PTBC() = delete; // default constructor is not allowed

  PTBC(const PTBCParams &params_, const index_t initial_index_,
       HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field_,
       std::shared_ptr<Integrator> integrator, RNG &rng_,
       std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : initial_index(initial_index_), current_index(initial_index_),
        params(params_), rng(rng_), dist(dist_), mt(mt_),
        hamiltonian_field(hamiltonian_field_),
        hmc(params_.hmc_params, hamiltonian_field_, integrator, rng_, dist_,
            mt_) {
    device_id = Kokkos::device_id(); // default device id

    // host only fallback
    if (device_id == -1) {
      device_id = 0;
      partner_device_id = 0; // default to first two devices
    } else {
      partner_device_id =
          (device_id + 1) % Kokkos::num_devices(); // default partner device id
    }

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
int run_PTBC(PTBCParams ptbc_params, RNG &rng,
             std::uniform_real_distribution<real_t> dist, std::mt19937 mt) {

  static_assert(DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind ==
                GaugeFieldKind::PTBC);
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
  HMCParams &hmc_params = ptbc_params.hmc_params;

  defectParams<Nd> dParams;
  dParams.defect_length = ptbc_params.defect_size;
  dParams.defect_value = ptbc_params.defect_value;

  HField hamiltonian_field = HField([&]() -> HField {
    if constexpr (Nd == 2) {

      if (hmc_params.coldStart) {
        typename DGaugeFieldType::type gauge_field(hmc_params.L0, hmc_params.L1,
                                                   identitySUN<Nc>(), dParams);
        typename DAdjFieldType::type adjoint_field(hmc_params.L0, hmc_params.L1,
                                                   traceT(identitySUN<Nc>()));

        return HField(gauge_field, adjoint_field);
      } else {
        typename DGaugeFieldType::type gauge_field(
            hmc_params.L0, hmc_params.L1, rng, hmc_params.rngDelta, dParams);
        typename DAdjFieldType::type adjoint_field(hmc_params.L0, hmc_params.L1,
                                                   traceT(identitySUN<Nc>()));

        return HField(gauge_field, adjoint_field);
      }

    } else if constexpr (Nd == 3) {

      if (hmc_params.coldStart) {
        typename DGaugeFieldType::type gauge_field(hmc_params.L0, hmc_params.L1,
                                                   hmc_params.L2,
                                                   identitySUN<Nc>(), dParams);
        typename DAdjFieldType::type adjoint_field(hmc_params.L0, hmc_params.L1,
                                                   hmc_params.L2,
                                                   traceT(identitySUN<Nc>()));

        return HField(gauge_field, adjoint_field);
      } else {
        typename DGaugeFieldType::type gauge_field(
            hmc_params.L0, hmc_params.L1, hmc_params.L2, rng,
            hmc_params.rngDelta, dParams);
        typename DAdjFieldType::type adjoint_field(hmc_params.L0, hmc_params.L1,
                                                   hmc_params.L2,
                                                   traceT(identitySUN<Nc>()));

        return HField(gauge_field, adjoint_field);
      }

    } else if constexpr (Nd == 4) {

      if (hmc_params.coldStart) {
        typename DGaugeFieldType::type gauge_field(hmc_params.L0, hmc_params.L1,
                                                   hmc_params.L2, hmc_params.L3,
                                                   identitySUN<Nc>(), dParams);
        typename DAdjFieldType::type adjoint_field(hmc_params.L0, hmc_params.L1,
                                                   hmc_params.L2, hmc_params.L3,
                                                   traceT(identitySUN<Nc>()));

        return HField(gauge_field, adjoint_field);
      } else {
        typename DGaugeFieldType::type gauge_field(
            hmc_params.L0, hmc_params.L1, hmc_params.L2, hmc_params.L3, rng,
            hmc_params.rngDelta, dParams);
        typename DAdjFieldType::type adjoint_field(hmc_params.L0, hmc_params.L1,
                                                   hmc_params.L2, hmc_params.L3,
                                                   traceT(identitySUN<Nc>()));

        return HField(gauge_field, adjoint_field);
      }

    } else {
      throw std::runtime_error("Invalid Nd");
    }
  }());

  using Update_Q = UpdatePositionGauge<Nd, Nc, GaugeFieldKind::PTBC>;
  using Update_P = UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>;

  Update_Q update_q(hamiltonian_field.gauge_field,
                    hamiltonian_field.adjoint_field);
  Update_P update_p(hamiltonian_field.gauge_field,
                    hamiltonian_field.adjoint_field, hmc_params.beta);
  // the integrate might need to be passed into the run_HMC as an argument as it
  // contains a large amount of design decisions
  std::shared_ptr<LeapFrog> leap_frog =
      std::make_shared<LeapFrog>(hmc_params.nstepsGauge, true, nullptr,
                                 std::make_shared<Update_Q>(update_q),
                                 std::make_shared<Update_P>(update_p));

  index_t initial_index{0};
  MPI_Comm_rank(MPI_COMM_WORLD, &initial_index);

  PTBC<DGaugeFieldType, DAdjFieldType, RNG> ptbc(
      ptbc_params, initial_index, hamiltonian_field, leap_frog, rng, dist, mt);

  Kokkos::Timer timer;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  index_t rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int swap_start;
  // MPI share first swap
  if (rank == 0) {
    swap_start = int(dist(mt) * (size + 1));
  }
  MPI_Bcast(&swap_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("Rank %d of %d, swap_start = %d\n", rank, size, swap_start);

  for (size_t step = 0; step < hmc_params.nsteps; ++step) {
    timer.reset();

    bool accept = ptbc.step();

    if (rank == 0) {
      swap_start = int(dist(mt) * (size + 1));
    }
    MPI_Bcast(&swap_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d of %d, swap_start = %d\n", rank, size, swap_start);

    // Gauge observables

    // PTBC swap/accept

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);
  }

  return 0;
}

#define INITIALIZE_PTBCPREPARE(ND, NC, RNG)                                    \
  template int run_PTBC<DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>,    \
                        DeviceAdjFieldType<ND, NC>, RNG>(                      \
      PTBCParams, RNG &, std::uniform_real_distribution<real_t>,               \
      std::mt19937);
INITIALIZE_PTBCPREPARE(2, 1, RNGType)
INITIALIZE_PTBCPREPARE(2, 2, RNGType)
// INITIALIZE_PTBCPREPARE(2, 3, RNGType)
INITIALIZE_PTBCPREPARE(3, 1, RNGType)
INITIALIZE_PTBCPREPARE(3, 2, RNGType)
// INITIALIZE_PTBCPREPARE(3, 3, RNGType)
INITIALIZE_PTBCPREPARE(4, 1, RNGType)
INITIALIZE_PTBCPREPARE(4, 2, RNGType)
// INITIALIZE_PTBCPREPARE(4, 3, RNGType)
#undef INITIALIZE_PTBCPREPARE
//

} // namespace klft
