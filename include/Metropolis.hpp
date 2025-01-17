#pragma once
#include "GaugeField.hpp"

namespace klft {

  template <typename T, class GaugeGroup, class GaugeFieldType, class RNG>
  class Metropolis {
    public:
      struct initGauge_cold_s {};
      struct initGauge_hot_s {};
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

      Kokkos::complex<T> get_det() const {
        Kokkos::complex<T> det = 1.0;
        auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_dim(0),gauge_field.get_dim(1),gauge_field.get_dim(2),gauge_field.get_dim(3),gauge_field.get_Ndim()}); 
        Kokkos::parallel_reduce("get_det", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu, Kokkos::complex<T> &det) {
          det += this->gauge_field.get_link(x,y,z,t,mu).det();
        }, det);
        Kokkos::fence();
        return det/(gauge_field.get_volume()*gauge_field.get_Ndim());
      }

      Kokkos::complex<T> check_gauge() const {
        Kokkos::complex<T> check = 0.0;
        auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_dim(0),gauge_field.get_dim(1),gauge_field.get_dim(2),gauge_field.get_dim(3),gauge_field.get_Ndim()});
        Kokkos::parallel_reduce("check_gauge", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu, Kokkos::complex<T> &check) {
          GaugeGroup c = dagger(this->gauge_field.get_link(x,y,z,t,mu))*this->gauge_field.get_link(x,y,z,t,mu);
          for(int i = 0; i < gauge_field.get_Nc(); i++) {
            check += c(i*gauge_field.get_Nc()+i);
          }
        }, check);
        Kokkos::fence();
        return check/(gauge_field.get_volume()*gauge_field.get_Ndim()*gauge_field.get_Nc());
      }

      KOKKOS_INLINE_FUNCTION void operator()(initGauge_cold_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        GaugeGroup U;
        U.set_identity();
        this->gauge_field.set_link(x,y,z,t,mu,U);
      }

      KOKKOS_INLINE_FUNCTION void operator()(initGauge_hot_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        auto generator = rng.get_state();
        GaugeGroup U;
        U.get_random(generator,delta);
        this->gauge_field.set_link(x,y,z,t,mu,U);
        rng.free_state(generator);
      }
      
      template <int odd_even>
      KOKKOS_INLINE_FUNCTION void operator()(sweep_s<odd_even>, const int &x, const int &y, const int &z, const int &t, const int &mu, T &update) const {
        if(t%2 == odd_even) return;
        auto generator = rng.get_state();
        T num_accepted = 0.0;
        T delS = 0.0;
        GaugeGroup U = gauge_field.get_link(x,y,z,t,mu);
        GaugeGroup staple = gauge_field.get_staple(x,y,z,t,mu);
        GaugeGroup tmp1 = U*staple;
        GaugeGroup R;
        for(int i = 0; i < n_hit; ++i) {
          R.get_random(generator, delta);
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

      void initGauge(const bool cold_start) {
        if(cold_start) {
          auto BulkPolicy = Kokkos::MDRangePolicy<initGauge_cold_s,Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_dim(0),gauge_field.get_dim(1),gauge_field.get_dim(2),gauge_field.get_dim(3),gauge_field.get_Ndim()});
          Kokkos::parallel_for("initGauge_cold", BulkPolicy, *this);
        } else {
          auto BulkPolicy = Kokkos::MDRangePolicy<initGauge_hot_s,Kokkos::Rank<5>>({0,0,0,0,0},{gauge_field.get_dim(0),gauge_field.get_dim(1),gauge_field.get_dim(2),gauge_field.get_dim(3),gauge_field.get_Ndim()});
          Kokkos::parallel_for("initGauge_hot", BulkPolicy, *this);
        }
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