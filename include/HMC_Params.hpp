#pragma once

namespace klft {

  class HMC_Params {
    public:
      size_t n_steps;
      double tau;
      HMC_Params() = default;
      HMC_Params(size_t _n_steps, double _tau) : n_steps(_n_steps), tau(_tau) {}

      double get_tau() const { return tau; }
      void set_tau(const double &_tau) { tau = _tau; }
      size_t get_n_steps() const { return n_steps; }
      void set_n_steps(const size_t &_n_steps) { n_steps = _n_steps; }
  };

} // namespace klft