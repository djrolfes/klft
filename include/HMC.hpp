#pragma once
#include "GLOBAL.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"

namespace klft {

  template<typename T, class Group, class Adjoint, class RNG, int Ndim = 4, int Nc = 3>
  class HMC {
  public:
    struct randomize_momentum_s {};
    HMC_Params params;
    HamiltonianField<T,Group,Adjoint,Ndim,Nc> hamiltonian_field;
    std::list<Monomial<T,Group,Adjoint,Ndim,Nc>> monomials;
    // Integrator<T,Group,Adjoint,Ndim,Nc> &integrator;
    RNG rng;

    HMC() = default;

    HMC(const HMC_Params _params, RNG _rng) : params(_params), rng(_rng) {}

    void add_monomial(Monomial<T,Group,Adjoint,Ndim,Nc> &monomial) {
      monomials.push_back(monomial);
    }

    void add_hamiltonian_field(const HamiltonianField<T,Group,Adjoint,Ndim,Nc> _hamiltonian_field) {
      hamiltonian_field = _hamiltonian_field;
    }

    // void set_integrator(Integrator<T,Group,Adjoint,Ndim,Nc> &_integrator) {
    //   integrator = _integrator;
    // }

    // KOKKOS_INLINE_FUNCTION void operator()(randomize_momentum_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
    //   auto generator = rng.get_state();
    //   Adjoint U;
    //   U.get_random(generator);
    //   hamiltonian_field.adjoint_field.set_adjoint(x,y,z,t,mu,U);
    //   rng.free_state(generator);
    // }

    // void randomize_momentum() {
    //   auto BulkPolicy = Kokkos::MDRangePolicy<randomize_momentum_s,Kokkos::Rank<5>>({0,0,0,0,0},{hamiltonian_field.adjoint_field.get_dim(0),hamiltonian_field.adjoint_field.get_dim(1),hamiltonian_field.adjoint_field.get_dim(2),hamiltonian_field.adjoint_field.get_dim(3),hamiltonian_field.adjoint_field.get_Ndim()});
    //   Kokkos::parallel_for("randomize_momentum", BulkPolicy, *this);
    // }

    // void hmc_step() {
    //   randomize_momentum();
    //   GaugeField<T,Group,Ndim,Nc> gauge_old(hamiltonian_field.gauge_field.get_dim(0),hamiltonian_field.gauge_field.get_dim(1),hamiltonian_field.gauge_field.get_dim(2),hamiltonian_field.gauge_field.get_dim(3));
    //   gauge_old.copy(hamiltonian_field.gauge_field);
    //   for(auto monomial : monomials) {
    //     monomial->heatbath(hamiltonian_field);
    //   }
    //   integrator.integrate(monomials, hamiltonian_field, params);
    //   T delta_H = 0.0;
    //   for(auto monomial : monomials) {
    //     monomial->accept();
    //     delta_H += monomial->get_delta_H();
    //   }
    //   bool accept = true;
    //   if(delta_H > 0.0) {
    //     if(rng() > exp(-delta_H)) {
    //       accept = false;
    //     }
    //   }
    //   if(!accept) {
    //     hamiltonian_field.gauge_field.copy(gauge_old);
    //   }
    // }

  };
} // namespace klft