#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <typename T, class GaugeGroup, int Ndim = 4, int Nc = 3>
  class GaugeField {
    public:
      struct set_one_s {};
      struct plaq_s {};
      using complex_t = Kokkos::complex<T>;
      using DeviceView = Kokkos::View<complex_t****>;
      // using HostView = typename DeviceView::HostMirror;

      DeviceView gauge[Ndim][Nc*Nc];
      // HostView gauge_host;
      
      int LT,LX,LY,LZ;
      Kokkos::Array<int,4> dims;

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
        this->dims = {LX,LY,LZ,LT};
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
        this->dims = {LX,LY,LZ,LT};
      }

      KOKKOS_FUNCTION int get_Ndim() const { return Ndim; }

      KOKKOS_FUNCTION int get_Nc() const { return Nc; }

      KOKKOS_FUNCTION size_t get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

      KOKKOS_FUNCTION size_t get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*Nc*Nc; }

      KOKKOS_FUNCTION int get_dim(const int &mu) const {
        return this->dims[mu];
      }

      KOKKOS_INLINE_FUNCTION void operator()(set_one_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        for(int i = 0; i < Nc*Nc; i++) {
          this->gauge[mu][i](x,y,z,t) = Kokkos::complex<T>(0.0,0.0);
        }
        for(int i = 0; i < Nc; i++) {
          this->gauge[mu][i*Nc+i](x,y,z,t) = Kokkos::complex<T>(1.0,0.0);
        }
      }

      void set_one() {
        auto BulkPolicy = Kokkos::MDRangePolicy<set_one_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->LX,this->LY,this->LZ,this->LT,Ndim});
        Kokkos::parallel_for("set_one", BulkPolicy, *this);
      }

      KOKKOS_INLINE_FUNCTION GaugeGroup get_link(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
        for(int i = 0; i < Nc*Nc; i++) {
          link[i] = this->gauge[mu][i](x,y,z,t);
        }
        return GaugeGroup(link);
      }

      KOKKOS_INLINE_FUNCTION GaugeGroup get_link(const Kokkos::Array<int,4> &site, const int &mu) const {
        Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
        for(int i = 0; i < Nc*Nc; i++) {
          link[i] = this->gauge[mu][i](site[0],site[1],site[2],site[3]);
        }
        return GaugeGroup(link);
      }

      KOKKOS_INLINE_FUNCTION void set_link(const int &x, const int &y, const int &z, const int &t, const int &mu, const GaugeGroup &U) const {
        for(int i = 0; i < Nc*Nc; i++) {
          this->gauge[mu][i](x,y,z,t) = U(i);
        }
      }

      KOKKOS_INLINE_FUNCTION void set_link(const Kokkos::Array<int,4> &site, const int &mu, const GaugeGroup &U) {
        for(int i = 0; i < Nc*Nc; i++) {
          this->gauge[mu][i](site[0],site[1],site[2],site[3]) = U(i);
        }
      }

      KOKKOS_INLINE_FUNCTION void operator()(plaq_s, const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq) const {
        GaugeGroup U1, U2, U3, U4;
        Kokkos::Array<int,4> site = {x,y,z,t};
        Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
        site_plus_mu[mu] = (site_plus_mu[mu] + 1) % this->dims[mu];
        for(int nu = 0; nu < mu; ++nu){
          Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
          site_plus_nu[nu] = (site_plus_nu[nu] + 1) % this->dims[nu];
          U1 = this->get_link(site,mu);
          U2 = this->get_link(site_plus_mu,nu);
          U3 = this->get_link(site_plus_nu,mu);
          U4 = this->get_link(site,nu);
          plaq += (U1*U2*dagger(U3)*dagger(U4)).retrace();
        }
      }

      T get_plaquette() {
        auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->LX,this->LY,this->LZ,this->LT,Ndim});
        T plaq = 0.0;
        Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
        return plaq/(this->get_volume()*((Ndim-1)*Ndim/2)*Nc);
      }

      KOKKOS_INLINE_FUNCTION GaugeGroup get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        GaugeGroup staple(0.0);
        GaugeGroup U1, U2, U3;
        Kokkos::Array<int,4> site = {x,y,z,t};
        Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
        site_plus_mu[mu] = (site_plus_mu[mu] + 1) % this->dims[mu];
        Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
        for(int nu = 0; nu < Ndim; ++nu) {
          if(nu == mu) continue;
          site_pm_nu[nu] = (site_pm_nu[nu] + 1) % this->dims[nu];
          U1 = get_link(site_plus_mu,nu);
          U2 = get_link(site_pm_nu,mu);
          U3 = get_link(site,nu);
          staple += U1*dagger(U2)*dagger(U3);
          site_pm_nu[nu] = (site_pm_nu[nu] - 1 + this->dims[nu]) % this->dims[nu];
        }
        for(int nu = 0; nu < Ndim; ++nu) {
          if(nu == mu) continue;
          site_plus_mu[nu] = (site_plus_mu[nu] - 1 + this->dims[nu]) % this->dims[nu];
          site_pm_nu[nu] = (site_pm_nu[nu] - 1 + this->dims[nu]) % this->dims[nu];
          U1 = get_link(site_plus_mu,nu);
          U2 = get_link(site_pm_nu,mu);
          U3 = get_link(site_pm_nu,nu);
          staple += dagger(U1)*dagger(U2)*U3;
          site_pm_nu[nu] = (site_pm_nu[nu] + 1) % this->dims[nu];
          site_plus_mu[nu] = (site_plus_mu[nu] + 1) % this->dims[nu];
        }
        return staple;
      }
  };

}
