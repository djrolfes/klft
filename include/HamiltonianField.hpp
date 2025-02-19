#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "AdjointField.hpp"
#include <iostream>

namespace klft {

  // Primary template: defaults to false.
template <typename>
struct is_kokkos_complex : std::false_type {};

// Specialization for Kokkos::complex.
template <typename T>
struct is_kokkos_complex<Kokkos::complex<T>> : std::true_type {};

  template<typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class HamiltonianField {
  public:
    struct update_momentum_s {};
    struct update_gauge_s {};
    GaugeField<T,Group,Ndim,Nc> gauge_field;
    AdjointField<T,Adjoint,Ndim,Nc> adjoint_field;

    typedef T value_type;
    
    HamiltonianField() = default;

    HamiltonianField(const GaugeField<T,Group,Ndim,Nc> _gauge_field, const AdjointField<T,Adjoint,Ndim,Nc> _adjoint_field) :
      gauge_field(_gauge_field), adjoint_field(_adjoint_field) {}
    
    //void update_momentum(AdjointField<T,Adjoint,Ndim,Nc> deriv, const T dtau) {
    //  auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{adjoint_field.get_max_dim(0),adjoint_field.get_max_dim(1),adjoint_field.get_max_dim(2),adjoint_field.get_max_dim(3),adjoint_field.get_Ndim()});
    //  Kokkos::parallel_for("update_momentum", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
    //    Adjoint tmp = adjoint_field.get_adjoint(x,y,z,t,mu);
    //    tmp -= dtau*deriv.get_adjoint(x,y,z,t,mu);
    //    adjoint_field.set_adjoint(x,y,z,t,mu,tmp);
    //  });
    //}

    //void update_gauge(const T dtau) {
    //  auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_max_dim(0),gauge_field.get_max_dim(1),gauge_field.get_max_dim(2),gauge_field.get_max_dim(3),gauge_field.get_Ndim()});
    //  Kokkos::parallel_for("update_gauge", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
    //    Group U = exp(dtau*adjoint_field.get_adjoint(x,y,z,t,mu))*gauge_field.get_link(x,y,z,t,mu);
    //    gauge_field.set_link(x,y,z,t,mu,U);
    //  });
    //}
void update_momentum(AdjointField<T,Adjoint,Ndim,Nc> deriv, const T dtau) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{adjoint_field.get_max_dim(0),adjoint_field.get_max_dim(1),adjoint_field.get_max_dim(2),adjoint_field.get_max_dim(3),adjoint_field.get_Ndim()});
      Kokkos::parallel_for("update_momentum", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
        
        
        Adjoint tmp = adjoint_field.get_adjoint(x,y,z,t,mu);
        
        
        tmp -= dtau*deriv.get_adjoint(x,y,z,t,mu);
        
        if (x == 1 && y == 1 && z == 1 && t == 1&&mu==1&&false) {
          constexpr int nLink = Group::nElements;    // e.g. 9 for SU3, 1 for U1
          constexpr int nAdj  = Adjoint::nElements;    // e.g. 8 for SU3, 1 for U1
          
          // Print the adjoint field dS:
          if constexpr(nAdj != 1) {
            Kokkos::printf("dtau: %f\n", dtau);
            for (int i = 0; i < nAdj; i++){
              Kokkos::printf("updateMomentumAdjoint (mu=%d): v[%d] = (%f);\n", mu, i, adjoint_field.get_adjoint(t,x,y,z,mu).v[i]);
            }
            Kokkos::printf("\n");
            for (int i = 0; i < nAdj; i++){
              Kokkos::printf("updateMomentumderiv (mu=%d): v[%d] = (%f);\n", mu, i, deriv.get_adjoint(t,x,y,z,mu).v[i]);
            }
            Kokkos::printf("\n");

            for (int i = 0; i < nAdj; i++){
              Kokkos::printf("updateMomentumRes (mu=%d): v[%d] = (%f);\n", mu, i, tmp.v[i]);
            }
            Kokkos::printf("\n");

            for (int i = 0; i < nAdj; i++){
              Kokkos::printf("wtf (mu=%d): v[%d] = (%f);\n", mu, i, adjoint_field.get_adjoint(x,y,z,t,mu).v[i]);
            }
            Kokkos::printf("\n");
          }

        }


        adjoint_field.set_adjoint(x,y,z,t,mu,tmp);

      });
    }


void update_gauge(const T dtau) {
  auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>(
    {0,0,0,0,0},
    {gauge_field.get_max_dim(0),
     gauge_field.get_max_dim(1),
     gauge_field.get_max_dim(2),
     gauge_field.get_max_dim(3),
     gauge_field.get_Ndim()}
  );
  Kokkos::parallel_for("update_gauge", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int t, const int x, const int y, const int z, const int mu) {
    Group link = gauge_field.get_link(t,x,y,z,mu);
    auto adj   = adjoint_field.get_adjoint(t,x,y,z,mu);  // e.g. AdjointSU3 or AdjointU1
    Group exp_val = exp(dtau * adj);
    Group U = exp_val * link;
    
    // Debug printing only for site (t,x,y,z) = (1,1,1,1)
    if (x == 1 && y == 1 && z == 1 && t == 1&&mu==1&&false) {
      constexpr int nLink = Group::nElements;    // e.g. 9 for SU3, 1 for U1
      constexpr int nAdj  = Adjoint::nElements;    // e.g. 8 for SU3, 1 for U1
      
      // Print the adjoint field dS:
      if constexpr(nAdj != 1) {
        for (int i = 0; i < nAdj; i++){
          Kokkos::printf("Adjoint (mu=%d): v[%d] = (%f);\n", mu, i, adjoint_field.get_adjoint(t,x,y,z,mu).v[i]);
        }
        Kokkos::printf("\n");
      }
      
      // Print the gauge link S:
      if constexpr(nLink != 1 && false) {
        for (int i = 0; i < nLink; i++){
          Kokkos::printf("Link (mu=%d): v[%d] = (%f, %f);\n", i, link.v[i].real(), mu,  link.v[i].imag());
        }
        Kokkos::printf("\n");
        for (int i = 0; i < nLink; i++){
          Kokkos::printf("exp_val (mu=%d): v[%d] = (%f, %f);\n", mu, i, exp_val.v[i]);
        }
        for (int i = 0; i < nLink; i++){
          Kokkos::printf("U (mu=%d): v[%d] = (%f, %f);\n", i, U.v[i].real(), mu,  U.v[i].imag());
        }
        Kokkos::printf("\n");
        Kokkos::printf("\n");
      }

    }
    gauge_field.set_link(t,x,y,z,mu,U);
  });
}


    T kinetic_energy() {
      T kinetic_energy = 0.0;
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{adjoint_field.get_max_dim(0),adjoint_field.get_max_dim(1),adjoint_field.get_max_dim(2),adjoint_field.get_max_dim(3),adjoint_field.get_Ndim()});
      Kokkos::parallel_reduce("kinetic_energy", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu, T &local_sum) {
        Adjoint tmp = adjoint_field.get_adjoint(x,y,z,t,mu);
        local_sum += 0.5*tmp.norm2();
      }, kinetic_energy);
      return kinetic_energy;
    }

    template <class RNG>
    void randomize_momentum(RNG rng) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{adjoint_field.get_max_dim(0),adjoint_field.get_max_dim(1),adjoint_field.get_max_dim(2),adjoint_field.get_max_dim(3),adjoint_field.get_Ndim()});
      Kokkos::parallel_for("randomize_momentum", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
        auto generator = rng.get_state();
        Adjoint U;
        U.get_random(generator);

        constexpr int nAdj  = Adjoint::nElements;    // e.g. 8 for SU3, 1 for U1
      
        // Print the adjoint field dS:
        if (x == 1 && y == 1 && z == 1 && t == 1 && mu==1){
        if constexpr(nAdj != 1) {
        for (int i = 0; i < nAdj; i++){
          Kokkos::printf("randAdjoint (mu=%d): v[%d] = (%f);\n", mu, i, U.v[i]);}
          Kokkos::printf("\n\n");
        }}

        adjoint_field.set_adjoint(x,y,z,t,mu,U);
        rng.free_state(generator);
      });
    }

  };

} // namespace klft