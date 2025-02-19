#pragma once
#include "GaugeMonomial.hpp"
#include "HamiltonianField.hpp"
#include "GaugeField.hpp"
#include "AdjointField.hpp"
#include "HMC_Params.hpp"

namespace klft {
  
  typedef enum IntegratorType_s {
    LEAPFROG = 0,
    LP_LEAPFROG
  } IntegratorType;

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Integrator {
  public:
    Integrator() {};
    virtual void integrate(std::vector<std::unique_ptr<Monomial<T,Group,Adjoint,Ndim,Nc>>> &monomials, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h, HMC_Params params) = 0;
  };

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Leapfrog : public Integrator<T,Group,Adjoint,Ndim,Nc> {
  public:
    Leapfrog() : Integrator<T,Group,Adjoint,Ndim,Nc>::Integrator() {}
    void integrate(std::vector<std::unique_ptr<Monomial<T,Group,Adjoint,Ndim,Nc>>> &monomials, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h, HMC_Params params) override {
      AdjointField<T,Adjoint,Ndim,Nc> deriv(h.gauge_field.dims);

      for (int mu = 0; mu < 4; ++mu) {
      for (int j = 0; j < deriv.adj_dim; ++j) {
        auto host_mirror = Kokkos::create_mirror_view(deriv.adjoint[mu][j]);
        Kokkos::deep_copy(host_mirror, deriv.adjoint[mu][j]);
        std::cout << "Dimensions of adjoint[" << mu << "][" << j << "]: "
                  << host_mirror.extent(0) << " x "
                  << host_mirror.extent(1) << " x "
                  << host_mirror.extent(2) << " x "
                  << host_mirror.extent(3) << std::endl;
  }
}

      T dtau = params.get_tau()/T(params.get_n_steps());
      // initial half step
      deriv.set_zero();
      Kokkos::fence();
      constexpr int nLink = Group::nElements;    // e.g. 9 for SU3, 1 for U1
      constexpr int nAdj  = Adjoint::nElements;    // e.g. 8 for SU3, 1 for U1
      if constexpr(nAdj != 1) {
        for (int i = 0; i < nAdj; i++){
          Kokkos::printf("derivAfterZeroAdjoint (mu=%d): v[%d] = (%f);\n", 1, i, deriv.get_adjoint(1,1,1,1,1).v[i]);
        }
        Kokkos::printf("\n");
      }


      for(int i = 0; i < monomials.size(); ++i) {
        if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      // full step for gauge
      h.update_gauge(dtau);
      // leapfrog steps
      for(size_t i = 0; i < params.get_n_steps(); ++i) {
        deriv.set_zero();
        Kokkos::fence();
        for(int i = 0; i < monomials.size(); ++i) {
          if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
        }
        h.update_momentum(deriv,dtau);
        h.update_gauge(dtau);
      }
      // final half step
      deriv.set_zero();
      Kokkos::fence();
      for(int i = 0; i < monomials.size(); ++i) {
        if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      h.gauge_field.restoreGauge();
    }
  };

} // namespace klft

