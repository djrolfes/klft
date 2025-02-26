#include "../include/klft.hpp"
#include "../include/HMC.hpp"


namespace klft {

  template<typename T, class Group, class Adjoint, class RNG, int Ndim = 4, int Nc = 3, int Nr = 3> // Nr = number of running hmcs
  class PTBC {
  public:
    //implement the parallel execution of Nr hmcs, I guess this could/should implement a PTBCHmcStep() function that allows for closer examination of the configs.

  }; // class PTBC
} //namespace klft
