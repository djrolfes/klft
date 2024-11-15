#pragma once
#include "GaugeField.hpp"

namespace klft {

  template <typename T, class GaugeGroup, class GaugeFieldType, class RNG>
  class Metropolis {
    public:
      template <int odd_even> struct sweep_s {};
      GaugeFieldType gauge_field;
      RNG rng;
      int n_hit;
      T beta;
      T delta;

      Metropolis() = default;

      Metropolis(GaugeFieldType gauge_field, RNG &rng, const int &n_hit, const T &beta, const T &delta) {
        this->gauge_field = gauge_field;
        this->rng = rng;
        this->n_hit = n_hit;
        this->beta = beta;
        this->delta = delta;
      }

      KOKKOS_FUNCTION T get_beta() const { return beta; }

      KOKKOS_FUNCTION T get_delta() const { return delta; }

      KOKKOS_FUNCTION T get_plaq() const { return gauge_field.get_plaquette(); }
      
      template <int odd_even>
      KOKKOS_INLINE_FUNCTION void operator()(sweep_s<odd_even>, const int &x, const int &y, const int &z, const int &t, const int &mu, T &update) const {
        if(t%2 == odd_even) return;
        auto generator = rng.get_state();
        T num_accepted = 0.0;
        T delS = 0.0;
        GaugeGroup U = gauge_field.get_link(x,y,z,t,mu);
        GaugeGroup staple = gauge_field.get_staple(x,y,z,t,mu);
        GaugeGroup tmp1 = U*staple;
        for(int i = 0; i < n_hit; ++i) {
          GaugeGroup R = get_random(generator, delta);
          GaugeGroup U_new = U*R;
          GaugeGroup tmp2 = U_new*staple;
          delS = (beta/static_cast<T>(gauge_field.get_Nc()))*(tmp1.retrace() - tmp2.retrace());
          bool accept = delS < 0.0;
          if(!accept) {
            T r = generator.drand(0.0,1.0);
            accept = r < Kokkos::exp(-delS);
          }
          if(accept) {
            U_new.restoreGauge();
            gauge_field.set_link(x,y,z,t,mu,U_new);
            num_accepted += 1.0;
          }
        }
        rng.free_state(generator);
        update += num_accepted;
      }

      T sweep() {
        auto BulkPolicy_odd = Kokkos::MDRangePolicy<sweep_s<1>,Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_dim(0),gauge_field.get_dim(1),gauge_field.get_dim(2),gauge_field.get_dim(3),gauge_field.get_Ndim()});
        auto BulkPolicy_even = Kokkos::MDRangePolicy<sweep_s<0>,Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_dim(0),gauge_field.get_dim(1),gauge_field.get_dim(2),gauge_field.get_dim(3),gauge_field.get_Ndim()});
        T accept = 0.0;
        T accept_rate = 0.0;
        Kokkos::parallel_reduce("sweep_even", BulkPolicy_even, *this, accept);
        Kokkos::fence();
        accept_rate += accept;
        accept = 0.0;
        Kokkos::parallel_reduce("sweep_odd", BulkPolicy_odd, *this, accept);
        Kokkos::fence();
        accept_rate += accept;
        return accept_rate/(gauge_field.get_volume()*gauge_field.get_Ndim()*n_hit);
      }

  };
}