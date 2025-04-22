#pragma once
#include "GLOBAL.hpp"

namespace klft {

  int Metropolis(const std::string &input_file);
                        
  template <typename T>
  void HMC_SU3_4D(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                  const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                  const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_SU3_3D(const size_t &LX, const size_t &LY, const size_t &LT,
                  const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                  const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_SU3_2D(const size_t &LX, const size_t &LT,
                  const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                  const size_t &seed, const std::string &outfilename);
}