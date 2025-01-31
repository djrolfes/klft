#pragma once
#include "GaugeMonomial.hpp"
#include "HamiltonianField.hpp"
#include "GaugeField.hpp"
#include "AdjointField.hpp"
#include "HMC_Params.hpp"

#include <list>

namespace klft {
  
  typedef enum IntegratorType_s {
    LEAPFROG = 0,
    LP_LEAPFROG
  } IntegratorType;

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Integrator {
  public:
    Integrator() {}
    virtual void integrate(std::list<Monomial<T,Group,Adjoint,Ndim,Nc>> monomials, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h, HMC_Params params) = 0;
  };

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Leapfrog : public Integrator<T,Group,Adjoint,Ndim,Nc> {
  public:
    Leapfrog() : Integrator<T,Group,Adjoint,Ndim,Nc>::Integrator() {}
    void integrate(std::list<Monomial<T,Group,Adjoint,Ndim,Nc>> monomials, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h, HMC_Params params) {
      AdjointField<T,Adjoint> deriv(h.gauge_field.get_dim(0),h.gauge_field.get_dim(1),h.gauge_field.get_dim(2),h.gauge_field.get_dim(3));
      T dtau = params.get_tau()/T(params.get_n_steps());
      // initial half step
      deriv.set_zero();
      for(auto monomial : monomials) {
        monomial->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      // full step for gauge
      h.update_gauge(dtau);
      // leapfrog steps
      for(size_t i = 0; i < params.get_n_steps(); ++i) {
        deriv.set_zero();
        for(auto monomial : monomials) {
          monomial->derivative(deriv,h);
        }
        h.update_momentum(deriv,dtau);
        h.update_gauge(dtau);
      }
      // final half step
      deriv.set_zero();
      for(auto monomial : monomials) {
        monomial->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      h.gauge_field.restoreGauge();
    }
  };

} // namespace klft

