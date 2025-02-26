#pragma once


namespace klft {
  template<typename T, int Ndim = 4>
  class PTBCDefect{
  public:
    T gauge_depression; //c(r) multiplied with links crossing the defect. TODO: make gauge_depression callable to allow for c(r) to depend on Lt-Lz
    size_t defect_length;
    size_t LX; // for rectangular lattices, the location/spacial boundary might be interesting as well.

    PTBCDefect() = default;

    PTBCDefect(const T &_gauge_depression, const size_t &_defect_length, const size_t &_LX){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
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
