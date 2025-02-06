#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <typename T, class Group, int Ndim = 4, int Nc = 3>
  class GaugeField {
  public:
    struct set_one_s {};
    struct plaq_s {};
    struct restoreGauge_s {};
    using complex_t = Kokkos::complex<T>;
    using DeviceView = Kokkos::View<complex_t****>;
    // using HostView = typename DeviceView::HostMirror;

    DeviceView gauge[Ndim][Nc*Nc];
    // HostView gauge_host;
    
    int LT,LX,LY,LZ;
    Kokkos::Array<int,Ndim> dims;
    Kokkos::Array<int,4> max_dims;
    Kokkos::Array<int,4> array_dims;


    typedef Group gauge_group_t;

    GaugeField() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeField(const int &_LX, const int &_LY, const int &_LZ, const int &_LT) {
      this->LT = _LT;
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeField(const Kokkos::Array<int,4> &_dims) {
      this->LT = _dims[0];
      this->LX = _dims[1];
      this->LY = _dims[2];
      this->LZ = _dims[3];
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeField(const int &_LX, const int &_LY, const int &_LT) {
      this->LT = _LT;
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeField(const Kokkos::Array<int,3> &_dims) {
      this->LT = _dims[0];
      this->LX = _dims[1];
      this->LY = _dims[2];
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeField(const int &_LX, const int &_LT) {
      this->LT = _LT;
      this->LX = _LX;
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeField(const Kokkos::Array<int,2> &_dims) {
      this->LT = _dims[0];
      this->LX = _dims[1];
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
    }

    KOKKOS_FUNCTION int get_Ndim() const { return Ndim; }

    KOKKOS_FUNCTION int get_Nc() const { return Nc; }

    KOKKOS_FUNCTION size_t get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

    KOKKOS_FUNCTION size_t get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*Nc*Nc; }

    KOKKOS_FUNCTION int get_dim(const int &mu) const {
      return this->dims[mu];
    }

    KOKKOS_FUNCTION int get_max_dim(const int &mu) const {
      return this->max_dims[mu];
    }

    KOKKOS_FUNCTION int get_array_dim(const int &mu) const {
      return this->array_dims[mu];
    }

    KOKKOS_INLINE_FUNCTION void operator()(set_one_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->gauge[mu][i](x,y,z,t) = Kokkos::complex<T>(0.0,0.0);
      }
      #pragma unroll
      for(int i = 0; i < Nc; i++) {
        this->gauge[mu][i*Nc+i](x,y,z,t) = Kokkos::complex<T>(1.0,0.0);
      }
    }

    void set_one() {
      auto BulkPolicy = Kokkos::MDRangePolicy<set_one_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("set_one", BulkPolicy, *this);
    }

    KOKKOS_INLINE_FUNCTION Group get_link(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        link[i] = this->gauge[mu][i](x,y,z,t);
      }
      return Group(link);
    }

    KOKKOS_INLINE_FUNCTION Group get_link(const Kokkos::Array<int,4> &site, const int &mu) const {
      Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        link[i] = this->gauge[mu][i](site[0],site[1],site[2],site[3]);
      }
      return Group(link);
    }

    KOKKOS_INLINE_FUNCTION void set_link(const int &x, const int &y, const int &z, const int &t, const int &mu, const Group &U) const {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->gauge[mu][i](x,y,z,t) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void set_link(const Kokkos::Array<int,4> &site, const int &mu, const Group &U) {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->gauge[mu][i](site[0],site[1],site[2],site[3]) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(plaq_s, const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq) const {
      Group U1, U2, U3, U4;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      #pragma unroll
      for(int nu = 0; nu < mu; ++nu){
        Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
        site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        U1 = this->get_link(site,mu);
        U2 = this->get_link(site_plus_mu,nu);
        U3 = this->get_link(site_plus_nu,mu);
        U4 = this->get_link(site,nu);
        plaq += (U1*U2*dagger(U3)*dagger(U4)).retrace();
      }
    }

    T get_plaquette(bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
      if(Normalize) plaq /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION Group get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group staple(0.0);
      Group U1, U2, U3;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        U1 = get_link(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site,nu);
        staple += U1*dagger(U2)*dagger(U3);
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        U1 = get_link(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site_pm_nu,nu);
        staple += dagger(U1)*dagger(U2)*U3;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] + 1) % this->dims[nu];
      }
      return staple;
    }

    void copy(const GaugeField<T,Group,Ndim,Nc> &in) {
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          Kokkos::deep_copy(this->gauge[mu][i], in.gauge[mu][i]);
        }
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(restoreGauge_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group tmp = this->get_link(x,y,z,t,mu);
      tmp.restoreGauge();
      this->set_link(x,y,z,t,mu,tmp);
    }

    void restoreGauge() {
      auto BulkPolicy = Kokkos::MDRangePolicy<restoreGauge_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("restoreGauge", BulkPolicy, *this);
    }

    template <class RNG>
    void set_random(T delta, RNG rng) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("set_random", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu) {
        auto generator = rng.get_state();
        Group U;
        U.get_random(generator,delta);
        this->set_link(x,y,z,t,mu,U);
        rng.free_state(generator);
      });
    }
    
  };

}
