#pragma once
#include "GaugeGroup.hpp"
#include "PTBCDefect.hpp"

namespace klft {

  template <typename T, class Group, int Ndim = 4, int Nc = 3>
  class GaugeField {
  public:
    struct set_one_s {};
    struct plaq_s {};
    struct restoreGauge_s {};
    struct topoCharge_s {};
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
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      this->LT = _LT;
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
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
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
      this->LX = _LX;
      this->LY = _LY;
      this->LT = _LT;
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
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LT = _dims[2];
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
      this->LX = _LX;
      this->LT = _LT;
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
      this->LX = _dims[0];
      this->LT = _dims[1];
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

    KOKKOS_INLINE_FUNCTION Group get_link_base_one(const Kokkos::Array<int,4> &site, const int &mu) const { 
      Kokkos::Array<int, 4> tmp = {site[0],site[1],site[2],site[3]};
      if (mu<0){
        tmp[-(mu)-1] = (tmp[-(mu)-1] - 1 + this->dims[-(mu)-1]) % this->dims[-(mu)-1];
        return dagger(this->get_link(tmp, -mu-1));
      }else{
        return this->get_link(tmp, mu-1);
      }
    }

    KOKKOS_INLINE_FUNCTION Group get_link_base_one(const int &x, const int &y, const int &z, const int &t, const int &mu) const { 
      Kokkos::Array<int, 4> tmp = {x,y,z,t};
      if (mu<0){
        tmp[-(mu)-1] = (tmp[-(mu)-1] - 1 + this->dims[-(mu)-1]) % this->dims[-(mu)-1];
        return dagger(this->get_link(tmp, -mu-1));
      }else{
        return this->get_link(tmp, mu-1);
      }
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

    KOKKOS_INLINE_FUNCTION T get_single_plaquette(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group U1, U2, U3, U4;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      T plaq {0.0};
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
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION T get_single_plaquette_1x2(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group U1, U2, U3, U4, U5, U6;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu_plus_mu = {x,y,z,t};
      
      T plaq {0.0};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      site_plus_mu_plus_mu[this->array_dims[mu]] = (site_plus_mu_plus_mu[this->array_dims[mu]] + 2) % this->dims[mu];
      #pragma unroll
      for(int nu = 0; nu < mu; ++nu){
        Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
        site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        Kokkos::Array<int,4> site_plus_nu_plus_mu = site_plus_nu;
        site_plus_nu_plus_mu[this->array_dims[nu]] = (site_plus_nu_plus_mu[this->array_dims[mu]] + 1) % this->dims[nu];

        
        U1 = this->get_link(site,mu);
        U2 = this->get_link(site_plus_mu,mu);
        U3 = this->get_link(site_plus_mu_plus_mu,nu);
        U4 = this->get_link(site_plus_nu_plus_mu, mu);
        U5 = this->get_link(site_plus_nu, mu);        
        U6 = this->get_link(site,nu);
        plaq += (U1*U2*U3*dagger(U4)*dagger(U5)*dagger(U6)).retrace();
      }
      return plaq;
    }

T get_plaquette_1x2(bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, KOKKOS_LAMBDA(const typename GaugeField<T,Group,Ndim,Nc>::plaq_s,
              const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq_i) {
        int x_index = mod(x, this->get_max_dim(0));
        int y_index = mod(y, this->get_max_dim(1));
        int z_index = mod(z, this->get_max_dim(2));
        int t_index = mod(t, this->get_max_dim(3));
        plaq_i += this->get_single_plaquette_1x2(x_index, y_index, z_index, t_index, mu);
      }, plaq);
      if(Normalize) plaq /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

 KOKKOS_INLINE_FUNCTION 
T get_single_plaquette(const int &x, const int &y, const int &z, const int &t, 
                         const int &mu, const PTBCDefect<T, Ndim> &defect) const {
  Group U1, U2, U3, U4;
  const Kokkos::Array<int,4> site = {x, y, z, t};
  Kokkos::Array<int,4> site_plus_mu = {x, y, z, t};
  T plaq {0.0};
  // Update site_plus_mu for the mu direction
  site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];

  #pragma unroll
  for (int nu = 0; nu < mu; ++nu) {
    Kokkos::Array<int,4> site_plus_nu = {x, y, z, t};
    site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1) % this->dims[nu];

    // Get the link elements (without the defect factors)
    Group link_site_mu = this->get_link(site, mu);
    Group link_site_plus_mu_nu = this->get_link(site_plus_mu, nu);
    Group link_site_plus_nu_mu = this->get_link(site_plus_nu, mu);
    Group link_site_nu = this->get_link(site, nu);
    // Now multiply by the defect factors:
    U1 = link_site_mu * defect(site, mu);
    U2 = link_site_plus_mu_nu * defect(site_plus_mu, nu);
    U3 = link_site_plus_nu_mu * defect(site_plus_nu, mu);
    U4 = link_site_nu * defect(site, nu);

    // Compute the plaquette contribution for this nu
    auto product = U1 * U2 * dagger(U3) * dagger(U4);
    T prod_retrace = product.retrace();
    plaq += prod_retrace;
  }
  return plaq;
}

    KOKKOS_INLINE_FUNCTION void operator()(plaq_s, const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq) const {
      plaq += this->get_single_plaquette(x,y,z,t,mu);
    }

    KOKKOS_INLINE_FUNCTION
    int mod(int a, int b) {
      int r = a % b;
      return (r < 0) ? r + b : r;
    }

    T get_plaquette_around_defect(PTBCDefect<T, Ndim> defect, bool Normalize = true) {
      //determine the plaquette of links of the defect as well as two lattice spacings away
      int x_min {-3}, x_max {1};
      int y_min {-2}, y_max {static_cast<int>(defect.defect_length) + 2}; //is this too much area covered?
      int z_min {-2}, z_max {static_cast<int>(defect.defect_length) + 2};
      int t_min {-2}, t_max {static_cast<int>(defect.defect_length) + 2};
      if (x_max - x_min > this->get_max_dim(0)){x_min = 0; x_max = this->get_max_dim(0);}
      if (y_max - y_min > this->get_max_dim(1)){y_min = 0; y_max = this->get_max_dim(1);}
      if (z_max - z_min > this->get_max_dim(2)){z_min = 0; z_max = this->get_max_dim(2);}
      if (t_max - t_min > this->get_max_dim(3)){t_min = 0; t_max = this->get_max_dim(3);}

      //Kokkos::printf("gauge_depression: %f\n", static_cast<double>(defect.gauge_depression));

      //auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{1,this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      using PolicyType = Kokkos::MDRangePolicy<plaq_s, Kokkos::Rank<5>, int>;
      PolicyType BulkPolicy({x_min, y_min, z_min, t_min, 0},
                  {x_max, y_max, z_max, t_max, Ndim});

      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, KOKKOS_LAMBDA(const typename GaugeField<T,Group,Ndim,Nc>::plaq_s,
              const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq_i) {
        int x_index = mod(x, this->get_max_dim(0));
        int y_index = mod(y, this->get_max_dim(1));
        int z_index = mod(z, this->get_max_dim(2));
        int t_index = mod(t, this->get_max_dim(3));
        plaq_i += this->get_single_plaquette(x_index, y_index, z_index, t_index, mu, defect);
      }
      , plaq);
      if(Normalize) plaq /= ((x_max-x_min)*(y_max-y_min)*(z_max-z_min)*(t_max-t_min))*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }
    
    T get_plaquette(bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
      if(Normalize) plaq /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

    T get_plaquette(PTBCDefect<T, Ndim> defect, bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, KOKKOS_LAMBDA(const typename GaugeField<T,Group,Ndim,Nc>::plaq_s,
              const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq_i) {
        int x_index = mod(x, this->get_max_dim(0));
        int y_index = mod(y, this->get_max_dim(1));
        int z_index = mod(z, this->get_max_dim(2));
        int t_index = mod(t, this->get_max_dim(3));
        plaq_i += this->get_single_plaquette(x_index, y_index, z_index, t_index, mu, defect);
      }, plaq);
      if(Normalize) plaq /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION int epsilon(const int &mu, const int &nu, const int &rho, const int &sigma) const {
      if (abs(mu)==abs(nu) || abs(mu)==abs(rho) || abs(mu)==abs(sigma) || abs(nu) == abs(rho) || abs(nu) == abs(sigma) || abs(rho) == abs(sigma)){return 0;} // this should never be entered I guess
      int rtn = 1;
      rtn *= mu/abs(mu) * nu/abs(nu) * rho/abs(rho) * sigma/abs(sigma);

      // get the number of inversions, this could and probably should be hardcoded (there are 12 odd permutations)
      int num_invs {0};
      int tmp[4]{abs(mu), abs(nu), abs(rho), abs(sigma)};
      #pragma unroll
      for (int i = 0; i<Ndim-1; ++i){
        #pragma unroll
        for (int j = i + 1; j < Ndim; ++j ){
          if (tmp[i]>tmp[j]){
            num_invs++;
            std::swap(tmp[i], tmp[j]);
          }
        }
      }

      if (num_invs%2 == 1){
        rtn *= -1;
      }
      return rtn;
    }

    KOKKOS_INLINE_FUNCTION T get_single_topological_charge_clover(const int &x, const int &y, const int &z, const int &t, const int &mu, const int &rho) const {
      // calculate the sum of clover traces at a site x, y, z, t
      if (mu == 0 || rho == 0 || abs(mu) == abs(rho)){return T(0.0);}
      
      T clov = T(0.0);
      Group U1, U2, U3, U4, tmpPlaq;
      const Kokkos::Array<int,4> site = {x, y, z, t};
      Kokkos::Array<int, 4> site_plus_mu = {x, y, z, t};
      site_plus_mu[this->array_dims[abs(mu)-1]] = (site_plus_mu[this->array_dims[abs(mu)-1]] + 1) % this->dims[abs(mu)-1];
      Kokkos::Array<int, 4> site_plus_tmp = {x, y, z, t};
      Kokkos::Array<int, 4> site_plus_rho = {x, y, z, t};
      site_plus_rho[this->array_dims[abs(rho)-1]] = (site_plus_rho[this->array_dims[abs(rho)-1]] + 1) % this->dims[abs(rho)-1];

      #pragma unroll
      for(int nu = -Ndim; nu < (Ndim+1); ++nu){
        if (abs(mu) == abs(nu) || abs(rho) == abs(nu) || nu == 0){continue;}
        site_plus_tmp[this->array_dims[abs(nu)-1]] = (site_plus_tmp[this->array_dims[abs(nu)-1]] + 1) % this->dims[abs(nu)-1];
        U1 = this->get_link_base_one(site,mu);
        U2 = this->get_link_base_one(site_plus_mu,nu);
        U3 = this->get_link_base_one(site_plus_tmp,mu);
        U4 = this->get_link_base_one(site,nu);
        tmpPlaq = (U1*U2*dagger(U3)*dagger(U4));
        site_plus_tmp[this->array_dims[abs(nu)-1]] = (site_plus_tmp[this->array_dims[abs(nu)-1]] - 1 + this->array_dims[abs(nu)-1]) % this->dims[abs(nu)-1];
        #pragma unroll
        for(int sigma = -Ndim; sigma <(Ndim+1); ++sigma){
          if (abs(mu) == abs(sigma) || abs(nu)==abs(sigma) || abs(rho) == abs(sigma) || sigma == 0){continue;}
          site_plus_tmp[this->array_dims[abs(sigma)-1]] = (site_plus_tmp[this->array_dims[abs(sigma)-1]] + 1) % this->dims[abs(sigma)-1];
          U1 = this->get_link_base_one(site,rho);
          U2 = this->get_link_base_one(site_plus_mu,sigma);
          U3 = this->get_link_base_one(site_plus_tmp,rho);
          U4 = this->get_link_base_one(site,sigma);
          clov += this->epsilon(mu, nu, rho, sigma)*(tmpPlaq*(U1*U2*dagger(U3)*dagger(U4))).retrace(); //is retrace correct here?
          site_plus_tmp[this->array_dims[abs(sigma)-1]] = (site_plus_tmp[this->array_dims[abs(sigma)-1]] - 1 + this->array_dims[abs(sigma)-1]) % this->dims[abs(sigma)-1];
        }       
      }
      return clov;
    }

    T get_topological_charge(bool Normalize = false){
      auto BulkPolicy = Kokkos::MDRangePolicy<topoCharge_s, Kokkos::Rank<6>, int>({0,0,0,0,-Ndim,-Ndim},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim+1, Ndim+1});
      T topoCharge = 0.0;
      Kokkos::parallel_reduce("topological charge", BulkPolicy, KOKKOS_LAMBDA(const typename GaugeField<T,Group,Ndim,Nc>::topoCharge_s,
        const int &x, const int &y, const int &z, const int &t, const int &mu, const int &rho, T &topoCharge_i){
          topoCharge_i += get_single_topological_charge_clover(x, y, z, t, mu, rho);
        }, topoCharge);
        topoCharge *= -1/(512*Kokkos::numbers::pi_v<T>*Kokkos::numbers::pi_v<T>);
        if(Normalize) topoCharge /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
        return topoCharge;
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
      #pragma unroll9
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

    KOKKOS_INLINE_FUNCTION Group get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu, const PTBCDefect<T, Ndim> &defect) const {
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
        U1 = get_link(site_plus_mu,nu)*defect(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu)*defect(site_pm_nu,mu);
        U3 = get_link(site,nu)*defect(site,nu);
        staple += U1*dagger(U2)*dagger(U3);
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        U1 = get_link(site_plus_mu,nu)*defect(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu)*defect(site_pm_nu,mu);
        U3 = get_link(site_pm_nu,nu)*defect(site_pm_nu,nu);
        staple += dagger(U1)*dagger(U2)*U3;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] + 1) % this->dims[nu];
      }
      return staple;
    }

    KOKKOS_INLINE_FUNCTION Group get_symanzik_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group staple(0.0);
      Group U1, U2, U3, U4, U5;
      Kokkos::Array<int,4> site1 = {x,y,z,t};
      Kokkos::Array<int,4> site2 = {x,y,z,t};
      Kokkos::Array<int,4> site3 = {x,y,z,t};
      Kokkos::Array<int,4> site4 = {x,y,z,t};
      Kokkos::Array<int,4> site5 = {x,y,z,t};
      site1[this->array_dims[mu]] = (site1[this->array_dims[mu]] + 1) % this->dims[mu];
      site2[this->array_dims[mu]] = (site2[this->array_dims[mu]] + 2) % this->dims[mu];
      site3[this->array_dims[mu]] = (site3[this->array_dims[mu]] + 1) % this->dims[mu];
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site4[this->array_dims[nu]] = (site4[this->array_dims[nu]] + 1) % this->dims[nu];
        site3[this->array_dims[nu]] = (site3[this->array_dims[nu]] + 1) % this->dims[nu];
        U1 = get_link(site1,mu);
        U2 = get_link(site2,nu);
        U3 = get_link(site3,mu);
        U4 = get_link(site4, mu);
        U5 = get_link(site5, nu);
        staple += U1*U2*dagger(U3)*dagger(U4)*dagger(U5);
        site4[this->array_dims[nu]] = (site4[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site3[this->array_dims[nu]] = (site3[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }
      //site1[this->array_dims[mu]] = (site1[this->array_dims[mu]] - 1) % this->dims[mu];
      site2[this->array_dims[mu]] = (site2[this->array_dims[mu]] - 2 + this->dims[mu]) % this->dims[mu];
      site3[this->array_dims[mu]] = (site3[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu];


      //site1[this->array_dims[mu]] = (site1[this->array_dims[mu]] + 1 + this->dims[mu]) % this->dims[mu];
      site3[this->array_dims[mu]] = (site3[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu];
      site4[this->array_dims[mu]] = (site4[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu];

      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site2[this->array_dims[nu]] = (site2[this->array_dims[nu]] + 1 + this->dims[nu]) % this->dims[nu];
        site3[this->array_dims[nu]] = (site3[this->array_dims[nu]] + 1 + this->dims[nu]) % this->dims[nu];
        U1 = get_link(site1, nu);
        U2 = get_link(site2, mu);
        U3 = get_link(site3, mu);
        U4 = get_link(site4, nu);
        U5 = get_link(site4, mu);
        staple += U1*dagger(U2)*dagger(U3)*dagger(U4)*U5;
        site2[this->array_dims[nu]] = (site2[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site3[this->array_dims[nu]] = (site3[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];

      }

      //site1[this->array_dims[mu]] = (site1[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu];
      site3[this->array_dims[mu]] = (site3[this->array_dims[mu]] + 1 + this->dims[mu]) % this->dims[mu];
      site4[this->array_dims[mu]] = (site4[this->array_dims[mu]] + 1 + this->dims[mu]) % this->dims[mu];

//---

      //site1[this->array_dims[mu]] = (site1[this->array_dims[mu]] + 1 + this->dims[mu]) % this->dims[mu];
      site2[this->array_dims[mu]] = (site2[this->array_dims[mu]] + 1 + this->dims[mu]) % this->dims[mu];


      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site1[this->array_dims[nu]] = (site1[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site2[this->array_dims[nu]] = (site2[this->array_dims[nu]] - 2 + this->dims[nu]) % this->dims[nu];
        site3[this->array_dims[nu]] = (site3[this->array_dims[nu]] - 2 + this->dims[nu]) % this->dims[nu];
        site4[this->array_dims[nu]] = (site4[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        U1 = get_link(site1, nu);
        U2 = get_link(site2, nu);
        U3 = get_link(site3, mu);
        U4 = get_link(site3, nu);
        U5 = get_link(site4, nu);
        staple += dagger(U1)*dagger(U2)*dagger(U3)*U4*U5;
        site1[this->array_dims[nu]] = (site1[this->array_dims[nu]] + 1 + this->dims[nu]) % this->dims[nu];
        site2[this->array_dims[nu]] = (site2[this->array_dims[nu]] + 2 + this->dims[nu]) % this->dims[nu];
        site3[this->array_dims[nu]] = (site3[this->array_dims[nu]] + 2 + this->dims[nu]) % this->dims[nu];
        site4[this->array_dims[nu]] = (site4[this->array_dims[nu]] + 1 + this->dims[nu]) % this->dims[nu];
      }

      //site1[this->array_dims[mu]] = (site1[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu];
      //site2[this->array_dims[mu]] = (site2[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu];

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
