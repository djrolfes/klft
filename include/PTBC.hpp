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
#include <sstream>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct PTBCParams {
  // Define parameters for the PTBC algorithm
  index_t n_sims;
  std::vector<real_t> defects; // a vector that hold the different defect values
  index_t defect_size;         // size of the defect on the lattice
  HMCParams hmc_params;        // HMC parameters´
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  real_t defect_value; // value of the defect
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

  std::vector<bool> swap_accepts; // a vector that hold the last values shown if
                                  // a given swap was accepted
  std::vector<real_t> swap_deltas; // a vector that holds the partial Delta_S
                                   // values for each swap
  int swap_start;                  // holds the rank of the last swap start

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
      : initial_index(initial_index_), params(params_), rng(rng_), dist(dist_),
        mt(mt_), hamiltonian_field(hamiltonian_field_),
        hmc(params_.hmc_params, hamiltonian_field_, integrator, rng_, dist_,
            mt_) {
    device_id = Kokkos::device_id(); // default device id
    current_index = initial_index_;

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

  void measure(GaugeObservableParams &gaugeObsParams, size_t step) {
    // measure the gauge observables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int compute_rank{0};
    int dummy_rank{0};

    if (rank == 0 || getDefectValue() >= 0.99999) {
      // only the computing rank or the defect rank measures the observables
      MPI_Reduce(
          &rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
          MPI_COMM_WORLD); // sum the ranks to get the computing rank in rank 0
      if (getDefectValue() >= 0.99999) {
        // if the defect value is close to 1, we measure the observables
        // for the PTBC case
        if (KLFT_VERBOSITY > 1) {
          printf("Measuring PTBC observables at step %zu\n", step);
        }
        measureGaugeObservablesPTBC<Nd, Nc>(hamiltonian_field.gauge_field,
                                            gaugeObsParams, step, 0, true);
      } else {

        measureGaugeObservablesPTBC<Nd, Nc>(hamiltonian_field.gauge_field,
                                            gaugeObsParams, step, compute_rank,
                                            false);
      }
    } else {
      MPI_Reduce(&dummy_rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
                 MPI_COMM_WORLD);
      return; // skip measurement for other ranks
    }
  }

  void measure(SimulationLoggingParams &simLogParams, index_t step,
               real_t acc_rate, bool accept, real_t time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
  }

  int step() {
    auto rtn = hmc.hmc_step();
    return rtn;
  }

  real_t swap_partner(index_t partner_rank) {
    // swaps the Defect with the partner rank and returns the partial Delta_S
    // MPI_Sendrecv(&current_index, 1, MPI_INT, partner_rank, TAG_INDEX,
    //              &current_index, 1, MPI_INT, partner_rank, TAG_INDEX,
    //              MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // DEBUG_MPI_PRINT("Swapping defect from index %d to %d", previous_index,
    //                 current_index);

    auto plaq = getPlaquetteAroundDefect();
    // DEBUG_MPI_PRINT("Plaquette around defect ss: %f", plaq);
    real_t S_ss = -(params.hmc_params.beta / static_cast<real_t>(Nc)) * plaq;
    // DEBUG_MPI_PRINT("S_ss = %f", S_ss);
    // now do the index/defect swapping
    hamiltonian_field.gauge_field.template set_defect<index_t>(
        params.defects[partner_rank]); // set the defect value of the current
    //
    plaq = getPlaquetteAroundDefect();
    // DEBUG_MPI_PRINT("Plaquette around defect sr: %f", plaq);
    real_t S_sr = -(params.hmc_params.beta / static_cast<real_t>(Nc)) * plaq;
    // DEBUG_MPI_PRINT("S_sr = %f", S_sr);
    return S_sr - S_ss; // return the partial Delta_S
  }

  void reverse_swap(bool accept) {
    if (!accept) {
      hamiltonian_field.gauge_field.template set_defect<index_t>(
          params.defects[current_index]); // set the defect value of the current
    }
  }

  void checkForNANs() {
    if constexpr (Nd == 4) {
#ifdef DEBUG_MPI
      // Create a host mirror and copy data from device to host
      auto host_field = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), hamiltonian_field.gauge_field.field);
      auto dimensions = hamiltonian_field.gauge_field.dimensions;

      for (index_t i = 0; i < dimensions[0]; ++i)
        for (index_t j = 0; j < dimensions[1]; ++j)
          for (index_t k = 0; k < dimensions[2]; ++k)
            for (index_t l = 0; l < dimensions[3]; ++l)
              for (index_t mu = 0; mu < Nd; ++mu) {
                auto link = host_field(i, j, k, l, mu);
                for (index_t c1 = 0; c1 < Nc; ++c1)
                  for (index_t c2 = 0; c2 < Nc; ++c2) {
                    if (std::isnan(Kokkos::real(link[c1][c2]))) {
                      DEBUG_MPI_PRINT("NaN at (%d,%d,%d,%d,mu=%d)", i, j, k, l,
                                      mu);
                    }
                  }
              }
#endif
    }
  }

  int swap() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int partner_rank;
    bool accept = false;
    real_t Delta_S{0};
    int swap_start{0};
    int swap_rank{0};

    // Rank 0 determines swap_start and broadcasts
    if (rank == 0) {
      swap_start = int(dist(mt) * (size));
      // DEBUG_MPI_PRINT("Rank 0 broadcasting swap_start = %d", swap_start);
    }

    MPI_Bcast(&swap_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // DEBUG_MPI_PRINT("Received broadcast swap_start = %d", swap_start);

    for (index_t i = 0; i < size; ++i) {
      swap_rank = (swap_start + i) % size;
      partner_rank = (swap_rank + 1) % size;

      // DEBUG_MPI_PRINT("Iteration %d: swap_rank=%d, partner_rank=%d", i,
      // swap_rank, partner_rank);

      // SWAP RANK sends its Delta_S
      if (rank == swap_rank) {
        real_t temp = swap_partner(partner_rank);
        // DEBUG_MPI_PRINT("Sending Delta_S_swap=%f to rank 0 (TAG_DELTAS)",
        // temp);
        MPI_Send(&temp, 1, mpi_real_t(), 0, TAG_DELTAS, MPI_COMM_WORLD);
      }

      // PARTNER RANK sends its Delta_S
      if (rank == partner_rank) {
        real_t temp = swap_partner(swap_rank);
        // DEBUG_MPI_PRINT("Sending Delta_S_partner=%f to rank 0
        // (TAG_DELTAS)",
        //                 temp);
        MPI_Send(&temp, 1, mpi_real_t(), 0, TAG_DELTAS, MPI_COMM_WORLD);
      }

      // RANK 0 gathers both values
      if (rank == 0) {
        real_t Delta_S_partner1, Delta_S_partner2;
        MPI_Recv(&Delta_S_partner1, 1, mpi_real_t(), swap_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // DEBUG_MPI_PRINT("Received Delta_S_swap=%f from swap_rank=%d",
        //                 Delta_S_partner1, swap_rank);

        MPI_Recv(&Delta_S_partner2, 1, mpi_real_t(), partner_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // DEBUG_MPI_PRINT("Received Delta_S_partner=%f from partner_rank=%d",
        //                 Delta_S_partner2, partner_rank);

        Delta_S = Delta_S_partner1 + Delta_S_partner2;

        // Decide accept/reject
        accept = true;
        if (Delta_S > 0) {
          if (dist(mt) > Kokkos::exp(-Delta_S)) {
            accept = false;
          }
        }
        // DEBUG_MPI_PRINT("Total Delta_S = %f, accept = %d", Delta_S,
        // accept);

        MPI_Send(&accept, 1, MPI_C_BOOL, swap_rank, TAG_ACCEPT, MPI_COMM_WORLD);
        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, TAG_ACCEPT,
                 MPI_COMM_WORLD);
        // DEBUG_MPI_PRINT("Sent accept=%d to swap_rank=%d and
        // partner_rank=%d",
        //                 accept, swap_rank, partner_rank);
      }

      // Swap ranks receive accept flag
      if (rank == swap_rank || rank == partner_rank) {
        // DEBUG_MPI_PRINT("Waiting to receive accept from rank 0");
        MPI_Recv(&accept, 1, MPI_C_BOOL, 0, TAG_ACCEPT, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        // DEBUG_MPI_PRINT("Received accept=%d from rank 0", accept);

        reverse_swap(accept);
        // DEBUG_MPI_PRINT("Reverse swap executed with accept=%d", accept);
      }

      // Rank 0 performs the swap if accepted
      if (rank == 0 && accept) {

        // DEBUG_MPI_PRINT("Swapping params.defects[%d] <->
        // params.defects[%d]",
        //                 swap_rank, partner_rank);
        std::swap(params.defects[swap_rank], params.defects[partner_rank]);
      }

      MPI_Barrier(MPI_COMM_WORLD); // synchronize all ranks after each swap
      MPI_Bcast(params.defects.data(), params.defects.size(), mpi_real_t(), 0,
                MPI_COMM_WORLD);
      if (rank == 0) {
        std::ostringstream oss;
        oss << "Defects after broadcast: [";
        for (size_t i = 0; i < params.defects.size(); ++i) {
          oss << params.defects[i];
          if (i + 1 < params.defects.size())
            oss << ", ";
        }
        oss << "]";
        DEBUG_MPI_PRINT("%s", oss.str().c_str());
      }
      MPI_Barrier(MPI_COMM_WORLD); // synchronize all ranks after each swap
    }

    // DEBUG_MPI_PRINT("Exiting swap() function");
    return 0;
  }

  real_t getPlaquetteAroundDefect() {
    // calculate the plaquette around the defect
    // this is a placeholder function, implement the actual plaquette
    // calculation
    auto plaq = GaugePlaquette<Nd, Nc, GaugeFieldKind::PTBC>(
        this->hamiltonian_field.gauge_field,
        false); // TODO: implement only calculating around the defect

    return plaq;
  }

private:
  int device_id;
  int partner_device_id;
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
  // the integrate might need to be passed into the run_HMC as an argument as
  // it contains a large amount of design decisions
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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (size_t step = 0; step < hmc_params.nsteps; ++step) {
    timer.reset();

    // DEBUG_MPI_PRINT("Enter ptbc(hmc) step: %zu", step);

    int accept = ptbc.step();
    MPI_Barrier(MPI_COMM_WORLD); // synchronize all ranks after step
    DEBUG_MPI_PRINT("HMC Step %zu: accept = %d", step, accept);

    int ptbc_accept = ptbc.swap();

    // Gauge observables
    ptbc.measure(ptbc_params.gaugeObsParams, step);

    if (rank == 0) {
      flushAllGaugeObservables(ptbc_params.gaugeObsParams, step, true);
    }
    // PTBC swap/accept

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);
    ptbc.measure(ptbc_params.simLogParams, step, acc_rate, accept, time);
    if (rank == 0) {
      Kokkos::printf("Step: %zu, accepted: %d, Acceptance rate: %f, Time: %f\n",
                     step, accept, acc_rate, time);
    }
    flushSimulationLogs(ptbc_params.simLogParams, step, true);
  }

  if (rank == 0) {
    forceflushAllGaugeObservables(ptbc_params.gaugeObsParams, true);
  }
  forceflushSimulationLogs(ptbc_params.simLogParams, true);

  return 0;
}

#define INITIALIZE_PTBCPREPARE(ND, NC, RNG)                                    \
  template int run_PTBC<DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>,    \
                        DeviceAdjFieldType<ND, NC>, RNG>(                      \
      PTBCParams, RNG &, std::uniform_real_distribution<real_t>,               \
      std::mt19937);
// INITIALIZE_PTBCPREPARE(2, 1, RNGType)
// INITIALIZE_PTBCPREPARE(2, 2, RNGType)
// // INITIALIZE_PTBCPREPARE(2, 3, RNGType)
// INITIALIZE_PTBCPREPARE(3, 1, RNGType)
// INITIALIZE_PTBCPREPARE(3, 2, RNGType)
// INITIALIZE_PTBCPREPARE(3, 3, RNGType)
INITIALIZE_PTBCPREPARE(4, 1, RNGType)
INITIALIZE_PTBCPREPARE(4, 2, RNGType)
// INITIALIZE_PTBCPREPARE(4, 3, RNGType)
#undef INITIALIZE_PTBCPREPARE
//

} // namespace klft
