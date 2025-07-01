#pragma once
#include "GLOBAL.hpp"
#include "GaugeMonomial.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include "Kokkos_Core_fwd.hpp"
#include "Monomial.hpp"
#include <random>

namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
class HMC {
public:
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeFieldType = typename DGaugeFieldType::type;
  using AdjFieldType = typename DAdjFieldType::type;

  struct randomize_momentum_s {};
  HMCParams params;
  HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field;
  std::vector<std::unique_ptr<Monomial<DGaugeFieldType, DAdjFieldType>>>
      monomials;
  std::shared_ptr<Integrator> integrator;
  RNG rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;
  real_t delta_H;
  GaugeFieldType gauge_old;

  HMC() = default;

  HMC(const HMCParams params_,
      HamiltonianField<DGaugeFieldType, DAdjFieldType> &hamiltonian_field_,
      std::shared_ptr<Integrator> integrator_, RNG rng_,
      std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : params(params_), rng(rng_), dist(dist_), mt(mt_),
        hamiltonian_field(hamiltonian_field_),
        integrator(std::move(integrator_)),
        gauge_old(hamiltonian_field_.gauge_field()) {}

  void add_gauge_monomial(const real_t _beta, const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<GaugeMonomial<DGaugeFieldType, DAdjFieldType>>(
            _beta, _time_scale));
  }

  void add_kinetic_monomial(const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<KineticMonomial<DGaugeFieldType, DAdjFieldType>>(
            _time_scale));
  }

  bool hmc_step() {

    hamiltonian_field.template randomize_momentum<RNG>(rng);

    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, gauge_old.field,
                      hamiltonian_field.gauge_field().field);
    Kokkos::DefaultExecutionSpace{}.fence();

    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->heatbath(hamiltonian_field);
    }
    integrator->integrate(params.tau, false);

    delta_H = 0.0;
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->accept(hamiltonian_field);
      real_t dH = monomials[i]->get_delta_H();
      delta_H += dH;
    }

    bool accept = true;
    if (delta_H > 0.0) {
      real_t rand = dist(mt);
      real_t threshold = Kokkos::exp(-delta_H);
      if (rand > threshold) {
        accept = false;
      }
    }

    if (!accept) {
      Kokkos::deep_copy(hamiltonian_field.gauge_field().field, gauge_old.field);
    } else {
    }

    return accept;
  }
};

} // namespace klft
