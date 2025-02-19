#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "Monomial.hpp"
#include "HamiltonianField.hpp"
#include <iostream>

namespace klft {

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class GaugeMonomial : public Monomial<T,Group,Adjoint,Ndim,Nc> {
    public:
      T beta;
    
    GaugeMonomial(T _beta, unsigned int _time_scale) : Monomial<T,Group,Adjoint,Ndim,Nc>::Monomial(_time_scale) {
      beta = _beta;
      Monomial<T,Group,Adjoint,Ndim,Nc>::monomial_type = KLFT_MONOMIAL_GAUGE;
    }

    void heatbath(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_old = -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false);
      std::cout << "H_old Gauge: " << -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false) << "\n";
    }

    void accept(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_new = -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false);
      std::cout << "H_new Gauge: " << -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false) << "\n";
    }

    void derivative(AdjointField<T,Adjoint,Ndim,Nc> deriv, 
                HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
                  constexpr int nAdj  = Adjoint::nElements;
                  if constexpr(nAdj != 1){
                  for (int i = 0; i < nAdj; i++){
          Kokkos::printf("derivBeforeParallel (mu=%d): v[%d] = (%f);\n", 1, i, deriv.adjoint[1][i](1,1,1,1));
        }}

  auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>(
      {0,0,0,0,0},
      { h.gauge_field.get_max_dim(0),
        h.gauge_field.get_max_dim(1),
        h.gauge_field.get_max_dim(2),
        h.gauge_field.get_max_dim(3),
        h.gauge_field.get_Ndim() } );
  
  Kokkos::parallel_for("derivative", BulkPolicy, KOKKOS_CLASS_LAMBDA(
      const int x, const int y, const int z, const int t, const int mu) {
    
    // Compute the staple S at this site and direction.
    Group S = h.gauge_field.get_staple(x,y,z,t,mu);
    S = h.gauge_field.get_link(x,y,z,t,mu) * S;
    if (x == 1 && y == 1 && z == 1 && t == 1 && mu== 1) {
      constexpr int nLink = Group::nElements;
      constexpr int nAdj  = Adjoint::nElements;   

      if constexpr(nAdj != 1) {
        for (int i = 0; i < nAdj; i++){
          Kokkos::printf("derivAdjoint (mu=%d): v[%d] = (%f);\n", mu, i, deriv.adjoint[mu][i](x,y,z,t));
        }}}
    // Construct an adjoint element from S.
    Adjoint dS(S);
    dS = (beta / h.gauge_field.get_Nc()) * dS;
    dS += deriv.get_adjoint(x,y,z,t,mu);
    
    // For debugging: if we are at a chosen site (say, x=y=z=t=1),
    // print all entries for each object on one line.
    if (x == 1 && y == 1 && z == 1 && t == 1 && mu==1 &&false) {
      constexpr int nLink = Group::nElements;    // e.g. 9 for SU3, 1 for U1
      constexpr int nAdj  = Adjoint::nElements;    // e.g. 8 for SU3, 1 for U1
      
      // Print the adjoint field dS:
      if constexpr(nAdj != 1) {
        for (int i = 0; i < nAdj; i++){
          Kokkos::printf("dSAdjoint (mu=%d): v[%d] = (%f);\n", mu, i, dS.v[i]);
        }
        Kokkos::printf("\n");
      }
      
      // Print the gauge link S:
      if constexpr(nLink != 1) {
        for (int i = 0; i < nLink; i++){
          Kokkos::printf("SLink (mu=%d): v[%d] = (%f, %f);\n", i, S.v[i].real(), mu,  S.v[i].imag());
        }
        Kokkos::printf("\n");
        Kokkos::printf("\n");
      }

    }
    
    deriv.set_adjoint(x,y,z,t,mu, dS);

    if (x == 1 && y == 1 && z == 1 && t == 1 && mu== 1) {
      constexpr int nLink = Group::nElements;
      constexpr int nAdj  = Adjoint::nElements;   

      if constexpr(nAdj != 1) {
        for (int i = 0; i < nAdj; i++){
          Kokkos::printf("derivAfterUpdateAdjoint (mu=%d): v[%d] = (%f);\n", mu, i, deriv.get_adjoint(x,y,z,t,mu).v[i]);
        }}}
  });
}

    //void derivative(AdjointField<T,Adjoint,Ndim,Nc> deriv, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
    //  auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{h.gauge_field.get_max_dim(0),h.gauge_field.get_max_dim(1),h.gauge_field.get_max_dim(2),h.gauge_field.get_max_dim(3),h.gauge_field.get_Ndim()});
    //  Kokkos::parallel_for("derivative", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu) {
    //    Group S = h.gauge_field.get_staple(x,y,z,t,mu);
    //    S = h.gauge_field.get_link(x,y,z,t,mu)*S;
    //    Adjoint dS(S);
    //    dS = (beta/h.gauge_field.get_Nc())*dS;
    //    dS += deriv.get_adjoint(x,y,z,t,mu);
    //    deriv.set_adjoint(x,y,z,t,mu,dS);
    //  });
    //}

  };

} // namespace klft