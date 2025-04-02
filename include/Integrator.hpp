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
      T dtau = params.get_tau()/T(params.get_n_steps());
      // initial half step
      deriv.set_zero();
      //Kokkos::printf("Before derivative\n");
      for(int i = 0; i < monomials.size(); ++i) {
        if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
      }
      //Kokkos::printf("After derivative\n");
      h.update_momentum(deriv,dtau/2.0);
      // full step for gauge
      h.update_gauge(dtau);
      //Kokkos::printf("E_kin: %f; ", h.kinetic_energy());
      //Kokkos::printf("H_W: %f; ", (T(-1.0)/T(h.gauge_field.get_Nc()))*(h.gauge_field.get_plaquette(false)));
      //Kokkos::printf("H: %f\n", (T(-1.0)/T(h.gauge_field.get_Nc()))*(T(5.0/6.0)*h.gauge_field.get_plaquette(false) - T(1.0/12.0)*h.gauge_field.get_plaquette_1x2(false)));
      // leapfrog steps
      for(size_t i = 0; i < params.get_n_steps(); ++i) {
        deriv.set_zero();
        for(int i = 0; i < monomials.size(); ++i) {
          if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
        }
        h.update_momentum(deriv,dtau);
        h.update_gauge(dtau);
        //Kokkos::printf("E_kin: %f; ", h.kinetic_energy());
        //Kokkos::printf("H_W: %f; ", (T(-1.0)/T(h.gauge_field.get_Nc()))*(h.gauge_field.get_plaquette(false)));
        //Kokkos::printf("H: %f\n", (T(-1.0)/T(h.gauge_field.get_Nc()))*(T(5.0/6.0)*h.gauge_field.get_plaquette(false) - T(1.0/12.0)*h.gauge_field.get_plaquette_1x2(false)));
      }
      // final half step
      deriv.set_zero();
      for(int i = 0; i < monomials.size(); ++i) {
        if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      h.gauge_field.restoreGauge();
    }
  };

} // namespace klft

