#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include <mpi.h>
#include <random>

namespace klft {

struct PTBCParams {
  // Define parameters for the PTBC algorithm
  index_t n_sims;
  std::vector<real_t> defects; // a vector that hold the different defect values
  index_t defect_size;         // size of the defect on the lattice
  const HMCParams hmc_params;  // HMC parameters´
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

  using GaugeFieldType = typename DGaugeFieldType::type;
  using AdjFieldType = typename DAdjFieldType::type;

public:
  PTBCParams params; // parameters for the PTBC algorithm
  HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field;
  HMC<DGaugeFieldType, DAdjFieldType, RNG> hmc;
  std::shared_ptr<Integrator> integrator;
  RNG rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;

  const index_t initial_index;
  index_t current_index;

  PTBC(const PTBCParams &params_, const index_t initial_index_,
       HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field_,
       std::shared_ptr<Integrator> integrator_, RNG rng_,
       std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : initial_index(initial_index_), current_index(initial_index_),
        params(params_), rng(rng_), dist(dist_), mt(mt_),
        hamiltonian_field(hamiltonian_field_),
        integrator(std::move(integrator_)) {
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
    hmc(params.hmc_params, hamiltonian_field, integrator, rng, dist, mt);
    hmc.add_gauge_monomial(params.hmc_params.beta, 0);
    hmc.add_kinetic_monomial(0);
  }

  real_t getDefectValue() const {
    // return the defect value for the current index
    return params.defects[current_index];
  }

  int step() {
    hmc.hmc_step();
    return 0;
  }

private:
  index_t device_id;
  index_t partner_device_id;
};
} // namespace klft
