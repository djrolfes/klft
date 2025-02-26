#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "Monomial.hpp"
#include "HamiltonianField.hpp"
#include "PTBCDefect.hpp"

namespace klft {
    //TODO: implement c(r)

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class GaugeMonomial : public Monomial<T,Group,Adjoint,Ndim,Nc> {
    public:
      T beta;
      const PTBCDefect defect;
    
    GaugeMonomial(T _beta, unsigned int _time_scale, const PTBCDefect _defect) : Monomial<T,Group,Adjoint,Ndim,Nc>::Monomial(_time_scale) {
      beta = _beta;
      Monomial<T,Group,Adjoint,Ndim,Nc>::monomial_type = KLFT_MONOMIAL_GAUGE;
      defect = _defect
    }

    void heatbath(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_old = -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false);
    }

    void accept(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_new = -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false);
    }

    void derivative(AdjointField<T,Adjoint,Ndim,Nc> deriv, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{h.gauge_field.get_max_dim(0),h.gauge_field.get_max_dim(1),h.gauge_field.get_max_dim(2),h.gauge_field.get_max_dim(3),h.gauge_field.get_Ndim()});
      Kokkos::parallel_for("derivative", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu) {
        Group S = h.gauge_field.get_staple_PTBC(x,y,z,t,mu, this->defect);
        S = this->defect(x,y,z,t,mu)*h.gauge_field.get_link(x,y,z,t,mu)*S;
        Adjoint dS(S);
        dS = (beta/h.gauge_field.get_Nc())*dS;
        dS += deriv.get_adjoint(x,y,z,t,mu);
        deriv.set_adjoint(x,y,z,t,mu,dS);
      });
    }

  };

} // namespace klft