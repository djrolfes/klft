#pragma once
#include "GLOBAL.hpp"
#include "GaugeMonomial.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
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
        gauge_old(hamiltonian_field.gauge_field.field, complex_t(0.0, 1.0)) {}

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
    hamiltonian_field.randomize_momentum(rng);
    Kokkos::printf("before gauge_old\n");

    for (int d = 0; d < 5; ++d) {
      Kokkos::printf(
          "extent[%d]: gauge_old = %d, hamiltonian = %d\n", d,
          static_cast<int>(gauge_old.field.extent(d)),
          static_cast<int>(hamiltonian_field.gauge_field.field.extent(d)));
    }
    Kokkos::fence();
    auto &dst = gauge_old.field;
    Kokkos::printf("after assign hfield\n");
    auto &src = hamiltonian_field.gauge_field.field;

    Kokkos::fence(); // flush prior work

    Kokkos::printf("COPYING GAUGE FIELD\n");
    Kokkos::printf("src: %d %d %d %d %d\n", src.extent(0), src.extent(1),
                   src.extent(2), src.extent(3), src.extent(4));
    Kokkos::printf("dst: %d %d %d %d %d\n ", dst.extent(0), dst.extent(1),
                   dst.extent(2), dst.extent(3), dst.extent(4));

    Kokkos::fence();             // flush prior work
    Kokkos::deep_copy(dst, src); // <--- crash point
    //     Kokkos::deep_copy(gauge_old.field,
    //     hamiltonian_field.gauge_field.field);
    Kokkos::printf("before Heatbath\n");
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->heatbath(hamiltonian_field);
    }
    Kokkos::printf("after Heatbath\n");
    integrator->integrate(params.tau, false);
    Kokkos::printf("after Integrate\n");
    delta_H = 0.0;
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->accept(hamiltonian_field);
      delta_H += monomials[i]->get_delta_H();
      // Kokkos::printf("delta_H: %f\n", delta_H);
    }
    bool accept = true;
    if (delta_H > 0.0) {
      if (dist(mt) > Kokkos::exp(-delta_H)) {
        accept = false;
      }
    }
    if (!accept) {
      Kokkos::deep_copy(hamiltonian_field.gauge_field.field, gauge_old.field);
    }
    return accept;
  }
};
} // namespace klft
