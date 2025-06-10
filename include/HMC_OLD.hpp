#pragma once
#include "GLOBAL.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include <random>

namespace klft {



  class HMC_Params_OLD {
    public:
      size_t n_steps;
      double tau;
      HMC_Params_OLD(size_t _n_steps, double _tau) : n_steps(_n_steps), tau(_tau) {}

      double get_tau() const { return tau; }
      void set_tau(const double &_tau) { tau = _tau; }
      size_t get_n_steps() const { return n_steps; }
      void set_n_steps(const size_t &_n_steps) { n_steps = _n_steps; }
  };

  template<typename T, class Group, class Adjoint, class RNG, int Ndim = 4, int Nc = 3>
  class HMC_OLD {
  public:
    struct randomize_momentum_s {};
    HMC_Params_OLD params;
    HamiltonianField<T,Group,Adjoint,Ndim,Nc> hamiltonian_field;
    std::vector<std::unique_ptr<Monomial<T,Group,Adjoint,Ndim,Nc>>> monomials;
    Integrator<T,Group,Adjoint,Ndim,Nc> *integrator;
    RNG rng;
    std::mt19937 mt;
    std::uniform_real_distribution<T> dist;

    HMC_OLD() = default;

    HMC_OLD(const HMC_Params_OLD _params, RNG _rng, std::uniform_real_distribution<T> _dist, std::mt19937 _mt) : params(_params), rng(_rng), dist(_dist), mt(_mt) {}

    void add_gauge_monomial(const T _beta, const unsigned int _time_scale) {
      monomials.emplace_back(std::make_unique<GaugeMonomial<T,Group,Adjoint,Ndim,Nc>>(_beta,_time_scale));
    }

    void add_kinetic_monomial(const unsigned int _time_scale) {
      monomials.emplace_back(std::make_unique<KineticMonomial<T,Group,Adjoint,Ndim,Nc>>(_time_scale));
    }

    void add_hamiltonian_field(const HamiltonianField<T,Group,Adjoint,Ndim,Nc> _hamiltonian_field) {
      hamiltonian_field = _hamiltonian_field;
    }

    void set_integrator(const IntegratorType _integrator_type) {
      switch(_integrator_type) {
        case LEAPFROG:
          integrator = new Leapfrog<T,Group,Adjoint,Ndim,Nc>();
          break;
        default:
          break;
      }
    }

    bool hmc_step() {
      hamiltonian_field.randomize_momentum(rng);
      GaugeField<T,Group,Ndim,Nc> gauge_old(hamiltonian_field.gauge_field.dims);
      gauge_old.copy(hamiltonian_field.gauge_field);
      for(int i = 0; i < monomials.size(); ++i) {
        monomials[i]->heatbath(hamiltonian_field);
      }
      integrator->integrate(monomials, hamiltonian_field, params);
      T delta_H = 0.0;
      for(int i = 0; i < monomials.size(); ++i) {
        monomials[i]->accept(hamiltonian_field);
        delta_H += monomials[i]->get_delta_H();
      }
      bool accept = true;
      if(delta_H > 0.0) {
        if(dist(mt) > Kokkos::exp(-delta_H)) {
          accept = false;
        }
      }
      if(!accept) {
        hamiltonian_field.gauge_field.copy(gauge_old);
      }
      return accept;
    }

  };
} // namespace klft
