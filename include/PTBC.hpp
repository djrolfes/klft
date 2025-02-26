#pragma once
#include "klft.hpp"
#include "HMC.hpp"
#include "PTBCDefect.hpp"
#include "PTBC_Params.hpp"


namespace klft {

  template<typename T, class Group, class Adjoint, class RNG, int Ndim = 4, int Nc = 2> // Nr = number of running hmcs
  class PTBC {
  public:
    using GaugeFieldType = GaugeField<T,Group,Ndim,Nc>;
    using AdjointFieldType = AdjointField<T,Adjoint,Ndim,Nc>;
    using HamiltonianFieldType = HamiltonianField<T,Group,Adjoint,Ndim,Nc>;
    struct randomize_momentum_s {};
    PTCB_Params ptcb_params;
    HMC_Params hmc_params;
    Integrator<T,Group,Adjoint,Ndim,Nc> *integrator;
    RNG rng;
    std::mt19937 mt;
    std::uniform_real_distribution<T> dist;
    std::vector<std::unique_ptr<HMC<T, Group, Adjoint, RNG, Ndim, Nc>>> hmcSims;
    std::vector<std::unique_ptr<PTBCDefect<T, Ndim>>> defects;


    PTBC() = default;

    PTBC(const PTBC_Params<T, Ndim> _ptcb_params, 
          const HMC_Params _hmc_params, RNG _rng, std::uniform_real_distribution<T> _dist, std::mt19937 _mt){
        ptcb_params(_ptcb_params);
        hmc_params(_hmc_params); 
        rng(_rng); 
        dist(_dist); 
        mt(_mt);
        this->init_HMCs();
        this->init_defects();
      }

    //implement the parallel execution of Nr hmcs, I guess this could/should implement a PTBCHmcStep() function that allows for closer examination of the configs.
    //for now, do not execute the hmc's in parallel for easier implementation TODO: parallelize
    void init_HMCs(){
      for (int i=0; i<ptcb_params.N_simulations;i++){
        hmcSims.emplace_back(std::make_unique<HMC<T,Group,Adjoint,RNG,Ndim,Nc>>(params,rng,dist,mt));
      }
    }

    T get_gauge_depression(int i){
      // returns the linearly interpolated gauge depression at lattice site i
      if (i == (N_simulations-1)){return T(0);}; //I don't think this is required.
      return T(1 - i/(ptcb_params.N_simulations-1));
    }

    void init_defects(){
      for (int i = 0; i< ptcb_params.N_simulations; i++){
        defects.emplace_back(std::make_unique<PTBCDefect<T, Ndim>>(get_gauge_depression(i), ptcb_params.defect_size, params, ptcb_params.LX));
      }
    }

    void add_gauge_monomials(const T _beta, const unsigned int _time_scale) {
      for (int i=0; i<ptcb_params.N_simulations; i++){
      hmcSims[i].add_gauge_monomial(_beta, _time_scale, defects[i]);
      }
    }

    void add_kinetic_monomials(const unsigned int _time_scale) {
      for (int i=0; i<ptcb_params.N_simulations; i++){
      hmcSims[i].add_kinetic_monomial(_timescale);
      }
    }

    void add_hamiltonian_fields(){
      for (int i = 0; i<ptcb_params.N_simulations; i++){
        GaugeFieldType gauge_field = GaugeFieldType(ptcb_params.get_lattice_dims());
        AdjointFieldType gauge_field = AdjointFieldType(ptcb_params.get_lattice_dims());
        HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
        hmcSims[i].add_hamiltonian_field(hamiltonian_field);
      }
    }

  }; // class PTBC
} //namespace klft
