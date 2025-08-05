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
  constexpr static size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((Nd == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  using Update_Q = UpdatePositionGauge<Nd, Nc, GaugeFieldKind::PTBC>;
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
  index_t previous_index;

  typedef enum {
    TAG_DELTAS = 0,
    TAG_ACCEPT = 1,
    TAG_INDEX = 2,
    TAG_SWAPSTART = 3
  } MPI_Tags;

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
    measureGaugeObservables<Nd, Nc>(hamiltonian_field.gauge_field,
                                    gaugeObsParams, step);
  }

  void measure(SimulationLoggingParams &simLogParams, index_t step,
               real_t acc_rate, bool accept, real_t time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
  }

  int step() { return hmc.hmc_step(); }

  real_t swap_rank(index_t rank) {
    // return the rank of the partner device for the given rank
  }

  real_t swap_partner(index_t partner_rank) {
    // swaps the Defect with the partner rank and returns the partial Delta_S
    previous_index = current_index;
    MPI_Sendrecv(&current_index, 1, MPI_INT, partner_rank, TAG_INDEX,
                 &current_index, 1, MPI_INT, partner_rank, TAG_INDEX,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    real_t S_ss = -params.hmc_params.beta / 3 * getPlaquetteAroundDefect();
    // now do the index/defect swapping
    hamiltonian_field.gauge_field.template set_defect<index_t>(
        params.defects[current_index]); // set the defect value of the current
    //
    real_t S_sr = -params.hmc_params.beta / 3 * getPlaquetteAroundDefect();
    return S_sr - S_ss; // return the partial Delta_S
  }

  void reverse_swap(bool accept) {
    if (!accept) {
      current_index = previous_index;
      hamiltonian_field.gauge_field.template set_defect<index_t>(
          params.defects[current_index]); // set the defect value of the current
    }
  }

  int swap() {
    // check if this PTBC instance is the first within the swap order, otherwise
    // wait for the previous one to contact it
    index_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t partner_rank;
    index_t partner_index;
    real_t partner_defect_value;
    bool accept = false;
    real_t Delta_S{0};
    int swap_start{0};
    index_t swap_rank{0};

    if (rank == 0) {
      swap_start = int(dist(mt) * (size + 1));
      MPI_Bcast(&swap_start, 1, MPI_INT, TAG_SWAPSTART, MPI_COMM_WORLD);
    } else {
      MPI_Bcast(&swap_start, 1, MPI_INT, TAG_SWAPSTART, MPI_COMM_WORLD);
    }
    for (index_t i = 0; i < size; ++i) {
      swap_rank = (swap_start + i) % size;
      partner_rank = (swap_rank + 1) % size;
      if (rank == swap_rank) {
        Delta_S = swap_partner(partner_rank);
        MPI_Send(&Delta_S, 1, MPI_REAL, 0, TAG_DELTAS, MPI_COMM_WORLD);
        MPI_Send(&current_index, 1, MPI_INT, 0, TAG_INDEX, MPI_COMM_WORLD);
      }
      if (rank == partner_rank) {
        Delta_S = swap_partner(swap_rank);
        MPI_Send(&Delta_S, 1, MPI_REAL, 0, TAG_DELTAS, MPI_COMM_WORLD);
        MPI_Send(&current_index, 1, MPI_INT, 0, TAG_INDEX, MPI_COMM_WORLD);
      }

      if (rank == 0) {
        // gather the Delta_S values from all ranks
        real_t Delta_S_partner1, Delta_S_partner2;
        MPI_Recv(&Delta_S_partner1, 1, MPI_REAL, swap_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&Delta_S_partner2, 1, MPI_REAL, partner_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        Delta_S = Delta_S_partner1 + Delta_S_partner2;
        accept = true; // accept the swap if Delta_S < 0
        if (Delta_S > 0) {
          if (dist(mt) > Kokkos::exp(-Delta_S)) {
            accept = false; // reject the swap
          }
        }
        MPI_Send(&accept, 1, MPI_C_BOOL, swap_rank, TAG_ACCEPT, MPI_COMM_WORLD);
        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, TAG_ACCEPT,
                 MPI_COMM_WORLD);
      }

      if (rank == swap_rank || rank == partner_rank) {
        // receive the accept value from rank 0
        MPI_Recv(&accept, 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        reverse_swap(accept); // reverse the swap if not accepted
      }

      if (rank == 0 && accept) {
        index_t swap_index, swap_partner_index;
        MPI_Recv(&swap_index, 1, MPI_INT, swap_rank, TAG_INDEX, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(&swap_partner_index, 1, MPI_INT, partner_rank, TAG_INDEX,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::swap(params.defects[swap_index],
                  params.defects[swap_partner_index]);
      }
      MPI_Bcast(params.defects.data(), params.defects.size(), MPI_REAL, 0,
                MPI_COMM_WORLD);
    }
  }

  real_t getPlaquetteAroundDefect() {
    // calculate the plaquette around the defect
    // this is a placeholder function, implement the actual plaquette
    // calculation
    return GaugePlaquette<Nd, Nc, GaugeFieldKind::PTBC>(
        hamiltonian_field
            .gauge_field); // TODO: implement only calculating around the defect
  }

private:
  index_t device_id;
  index_t partner_device_id;
};

// below: Functions used to dispatch the PTBC algorithm

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

  for (size_t step = 0; step < hmc_params.nsteps; ++step) {
    timer.reset();

    bool accept = ptbc.step();

    // maybe HMC Gauge Obeservables here
    //
    bool ptbc_accept = ptbc.swap();

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
