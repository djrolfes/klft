#pragma once
#include <random>

#include "AdjointFieldHelper.hpp"
#include "FermionMonomial.hpp"
#include "FermionParams.hpp"
#include "GLOBAL.hpp"
#include "GaugeMonomial.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include "Monomial.hpp"
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
  Integrator_Params params;
  HamiltonianField<DGaugeFieldType, DAdjFieldType>& hamiltonian_field;
  std::vector<std::unique_ptr<Monomial<DGaugeFieldType, DAdjFieldType>>>
      monomials;
  std::shared_ptr<Integrator> integrator;
  RNG rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;
  real_t delta_H;

  HMC() = default;

  HMC(const Integrator_Params params_,
      HamiltonianField<DGaugeFieldType, DAdjFieldType>& hamiltonian_field_,
      std::shared_ptr<Integrator> integrator_, RNG rng_,
      std::uniform_real_distribution<real_t> dist_, std::mt19937 mt_)
      : params(params_),
        rng(rng_),
        dist(dist_),
        mt(mt_),
        hamiltonian_field(hamiltonian_field_),
        integrator(std::move(integrator_)) {}

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
  template <typename DiracOperator, typename Solver, typename DSpinorFieldType>
  void add_fermion_monomial(
      typename DSpinorFieldType::type& spinorField,
      diracParams<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank,

                  DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim>&
          params_,
      const real_t& tol_, RNG& rng, const unsigned int _time_scale) {
    monomials.emplace_back(
        std::make_unique<
            FermionMonomial<DiracOperator, Solver, RNG, DSpinorFieldType,
                            DGaugeFieldType, DAdjFieldType>>(
            spinorField, params_, tol_, rng, _time_scale));
  }

  bool hmc_step(const bool& check_Reversibility = false) {
    Kokkos::fence();
    hamiltonian_field.randomize_momentum(rng);
    // print_SUNAdj(hamiltonian_field.adjoint_field(0, 0, 0, 0, 0),
    //              "Randomized Momentum");

    Kokkos::fence();
    GaugeFieldType gauge_old(hamiltonian_field.gauge_field.dimensions, rng,
                             0.1);
    Kokkos::deep_copy(gauge_old.field, hamiltonian_field.gauge_field.field);
    Kokkos::fence();
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->heatbath(hamiltonian_field);
    }
    integrator->integrate(params.tau, false);
    delta_H = 0.0;
    for (int i = 0; i < monomials.size(); ++i) {
      monomials[i]->accept(hamiltonian_field);
      delta_H += monomials[i]->get_delta_H();
      if (KLFT_VERBOSITY > 2) {
        monomials[i]->print();
      }

      // Kokkos::printf("delta_H_monomial: %.20f\n",
      // monomials[i]->get_delta_H());
    }
    // Kokkos::printf("delta_H_ges %.20f \n", delta_H);
    bool accept = true;
    if (delta_H > 0.0) {
      if (dist(mt) > Kokkos::exp(-delta_H)) {
        accept = false;
      }
    }
    if (check_Reversibility) {
      GaugeFieldType gauge_save(hamiltonian_field.gauge_field.dimensions, rng,
                                0.1);
      Kokkos::deep_copy(gauge_save.field, hamiltonian_field.gauge_field.field);
      Kokkos::fence();
      for (int i = 0; i < monomials.size(); ++i) {
        monomials[i]->reset();
        monomials[i]->heatbath(hamiltonian_field);
      }
      real_t delta_H_revers = 0;
      flip_sign<DAdjFieldType>(hamiltonian_field.adjoint_field);
      integrator->integrate(params.tau, false);
      for (int i = 0; i < monomials.size(); ++i) {
        monomials[i]->accept(hamiltonian_field);
        delta_H_revers += monomials[i]->get_delta_H();
        if (KLFT_VERBOSITY > 0) {
          printf("ReverseCheck Monomial\n");
          monomials[i]->print();
        }
      }
      Kokkos::printf("Deltadelta_H_ges %.20f \n", delta_H + delta_H_revers);
      Kokkos::deep_copy(hamiltonian_field.gauge_field.field, gauge_save.field);
    }
    if (!accept) {
      Kokkos::deep_copy(hamiltonian_field.gauge_field.field, gauge_old.field);
    }

    return accept;
  }
};
}  // namespace klft
