#pragma once
#include "GLOBAL.hpp"

namespace klft {

  template<typename T, int Ndim = 4>
  class PTBC_Params {
    public:
    // Do I want to use this to save all simulation parameters for the PTBC?
    size_t N_simulations;
    size_t defect_size;
    size_t LX, LY, LZ, LT;
    size_t seed;
    
    PTBC_Params() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    PTBC_Params(const size_t &_N_simulations, const size_t &_defect_size, const size_t &_seed, const size_t &_LX,const size_t &_LY, const size_t &_LZ,const size_t &_LT) {
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      this->LT = _LT;
      this->N_simulations = _N_simulations;
      this->defect_size = _defect_size;
      this->seed = _seed;
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    PTBC_Params(const size_t &_N_simulations, const size_t &_defect_size, const size_t &_seed, const size_t &_LX,const size_t &_LY,const size_t &_LT) {
      this->LX = _LX;
      this->LY = _LY;
      this->LT = _LT;
      this->LZ = 1;
      this->N_simulations = _N_simulations;
      this->defect_size = _defect_size;
      this->seed = _seed;
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    PTBC_Params(const size_t &_N_simulations, const size_t &_defect_size, const size_t &_seed, const size_t &_LX,const size_t &_LT) {
      this->LX = _LX;
      this->LT = _LT;
      this->LY = 1;
      this->LZ = 1;
      this->N_simulations = _N_simulations;
      this->defect_size = _defect_size;
      this->seed = _seed;
    }

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    KOKKOS_INLINE_FUNCTION Kokkos::Array<int,4> get_lattice_dims() const{
        return Kokkos::Array<int, 4> {static_cast<int>(LX), static_cast<int>(LY), static_cast<int>(LZ), static_cast<int>(LT)};
    }
    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    KOKKOS_INLINE_FUNCTION Kokkos::Array<int,Ndim> get_lattice_dims() const{
        return Kokkos::Array<int, Ndim> {static_cast<int>(LX), static_cast<int>(LY), static_cast<int>(LT)};
    }
    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    KOKKOS_INLINE_FUNCTION Kokkos::Array<int,Ndim> get_lattice_dims() const{
        return Kokkos::Array<int, Ndim> {static_cast<int>(LX), static_cast<int>(LT)};
    }

  };//class PTBC_Params 
}//namespace klft