#pragma once


namespace klft {
  template<typename T, int Ndim = 4>
  class PTBCDefect{
  public:
    double gauge_depression; //c(r) multiplied with links crossing the defect. TODO: make gauge_depression callable to allow for c(r) to depend on Lt-Lz
    int defect_length;
    int LX; // for rectangular lattices, the location/spacial boundary might be interesting as well.

    PTBCDefect(const double &_gauge_depression, const int &_defect_length, const int &_LX){
      this->gauge_depression = _gauge_depression;
      this->defect_length = _defect_length;
      this->LX = _LX;
    }



    KOKKOS_INLINE_FUNCTION double operator()(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      if (unlikely(mu==1 
          && x == this->LX-1
          && y < this->defect_length
          && z < this->defect_length
          && t < this->defect_lenght)){
            return this->gauge_depression;
          }
      return 1.0
    }

    KOKKOS_INLINE_FUNCTION double operator()(const Kokkos::Array<int,4> &site, const int &mu) const {
      if (unlikely(mu==1 
          && site[0] == this->LX-1
          && site[1] < this->defect_length
          && site[2] < this->defect_length
          && site[3] < this->defect_lenght)){
            return this->gauge_depression;
          }
      return 1.0
    }

  };//class PTBCDefect
}//namespace klft
