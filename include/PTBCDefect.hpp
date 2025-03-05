#pragma once


namespace klft {
  template<typename T, int Ndim = 4>
  class PTBCDefect{
  public:
    T gauge_depression; //c(r) multiplied with links crossing the defect. TODO: make gauge_depression callable to allow for c(r) to depend on Lt-Lz
    size_t defect_length;
    size_t LX, LY, LZ, LT; // for rectangular lattices, the location/spacial boundary might be interesting as well.
    Kokkos::Array<size_t,Ndim> dims; // TODO: think about a lattice params class
    Kokkos::Array<size_t, Ndim> array_dims;

    PTBCDefect() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
      this->LY = _LX;
      this->LZ = _LX;
      this->LT = _LX;
      this->dims = {LX, LY, LZ, LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX, const size_t &_LY, const size_t &_LZ, const size_t &_LT){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      this->LT = _LT;
      this->dims = {LX, LY, LZ, LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const Kokkos::Array<int,4> &_dims){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
      this->dims = {LX, LY, LZ, LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
      this->LY = _LX;
      this->LZ = 1;
      this->LT = _LX;
      this->dims = {LX, LY, LZ, LT};
      this->array_dims = {0,1,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX, const size_t &_LY, const size_t &_LT){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = 1;
      this->LT = _LT; 
      this->dims = {LX, LY, LT};
      this->array_dims = {0,1,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const Kokkos::Array<int,3> &_dims){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = 1;
      this->LT = _dims[2];
      this->dims = {LX, LY, LT};
      this->array_dims = {0,1,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX, const size_t &_LT){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
      this->LY = 1;
      this->LZ = 1;
      this->LT = _LX;
      this->dims = {LX, LT};
      this->array_dims = {0,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
      this->LY = 1;
      this->LZ = 1;
      this->LT = _LX;
      this->dims = {LX, LY, LZ, LT};
      this->array_dims = {0,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const Kokkos::Array<int,2> &_dims){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _dims[0];
      this->LY = 1;
      this->LZ = 1;
      this->LT = _dims[1];
      this->dims = {LX, LT};
      this->array_dims = {0,3};
    }

    KOKKOS_INLINE_FUNCTION T get_plaquette_depression(Kokkos::Array<int, 4> &site, const int &mu, const int &nu){
      // returns the product of the gauge depressions of a given plaquette
      if (mu!=1 && nu!=1){return T(1);}
      Kokkos::Array<int,4> site_plus_mu = site;
      Kokkos::Array<int,4> site_plus_nu = site;
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1) % this->dims[nu];
      return this->operator()(site,mu)*this->operator()(site_plus_mu,nu)*this->operator()(site_plus_nu,mu)*this->operator()(site,nu);
    }

     KOKKOS_INLINE_FUNCTION T get_plaquette_depression(const int &x, const int &y, const int &z, const int &t, const int &mu, const int &nu){
      // returns the product of the gauge depressions of a given plaquette
      Kokkos::Array<int,4> site = {x,y,z,t};
      return this->get_plaquette_depression(site, mu, nu);
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      if ((mu==1 
          && x == this->LX-1
          && y < defect_length
          && z < defect_length
          && t < defect_length)){
            return this->gauge_depression;
          }
      return T(1);
    }

    KOKKOS_INLINE_FUNCTION T operator()(const Kokkos::Array<int,4> &site, const int &mu) const {
      if ((mu==1 
          && site[0] == this->LX-1
          && site[1] < defect_length
          && site[2] < defect_length
          && site[3] < defect_length)){
            return this->gauge_depression;
          }
      return T(1);
    }

  };//class PTBCDefect
}//namespace klft
