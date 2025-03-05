#pragma once
#include "klft.hpp"
#include "HMC.hpp"
#include "PTBCDefect.hpp"
#include "PTBC_Params.hpp"
#include <iostream>


namespace klft {

  template<typename T, class Group, class Adjoint, class RNG, int Ndim = 4, int Nc = 2> // Nr = number of running hmcs
  class PTBC {
  public:
    using GaugeFieldType = GaugeField<T,Group,Ndim,Nc>;
    using AdjointFieldType = AdjointField<T,Adjoint,Ndim,Nc>;
    using HamiltonianFieldType = HamiltonianField<T,Group,Adjoint,Ndim,Nc>;
    struct randomize_momentum_s {};
    PTBC_Params<T, Ndim> ptcb_params;
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
        this->ptcb_params = _ptcb_params;
        this->hmc_params = _hmc_params; 
        this->rng = _rng; 
        this->dist = _dist; 
        this->mt = _mt;
        this->init_HMCs();
        this->init_defects();
        this->add_hamiltonian_fields();
      }

    //implement the parallel execution of Nr hmcs, I guess this could/should implement a PTBCHmcStep() function that allows for closer examination of the configs.
    //for now, do not execute the hmc's in parallel for easier implementation TODO: parallelize
    void init_HMCs(){
      for (int i=0; i<ptcb_params.N_simulations;i++){
        hmcSims.emplace_back(std::make_unique<HMC<T,Group,Adjoint,RNG,Ndim,Nc>>(hmc_params,rng,dist,mt));
      }
    }

    void set_integrators(const IntegratorType _integrator_type){
      for (int i=0; i<ptcb_params.N_simulations; ++i){
        hmcSims[i]->set_integrator(_integrator_type);
      }
    }

    T get_gauge_depression(int i){
      // returns the linearly interpolated gauge depression at lattice site i
      if (i == (ptcb_params.N_simulations-1)){return T(0);}; //I don't think this is required.
      return T(1 - i/(ptcb_params.N_simulations-1));
    }

    void init_defects(){
      for (int i = 0; i< ptcb_params.N_simulations; i++){
        defects.emplace_back(std::make_unique<PTBCDefect<T, Ndim>>(get_gauge_depression(i), ptcb_params.defect_size, ptcb_params.LX));
      }
    }

    void add_hamiltonian_fields(){
      for (int i = 0; i<ptcb_params.N_simulations; i++){
        GaugeFieldType gauge_field = GaugeFieldType(ptcb_params.get_lattice_dims());
        AdjointFieldType adjoint_field = AdjointFieldType(ptcb_params.get_lattice_dims());
        HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
        hmcSims[i]->add_hamiltonian_field(hamiltonian_field);
      }
    }

    void add_gauge_monomials(const T _beta, const unsigned int _time_scale) {
      for (int i=0; i<ptcb_params.N_simulations; i++){
      hmcSims[i]->add_gauge_monomial(_beta, _time_scale, *defects[i]);
      }
    }

    void add_kinetic_monomials(const unsigned int _time_scale) {
      for (int i=0; i<ptcb_params.N_simulations; i++){
      hmcSims[i]->add_kinetic_monomial(_time_scale);
      }
    }

    void swap_areas(int r, int s){
      //swaps the areas defined by the defect in hmcSims[r] with hmcSims[s]

      using complex_t = Kokkos::complex<T>;
      using DeviceView = Kokkos::View<complex_t****>;

      auto& gauge_r = hmcSims[r]->hamiltonian_field.gauge_field.gauge;
      auto& gauge_s = hmcSims[s]->hamiltonian_field.gauge_field.gauge;

      int y_end = std::min(ptcb_params.LY, ptcb_params.defect_size); // this might need a -1
      int z_end = std::min(ptcb_params.LZ, ptcb_params.defect_size);
      int t_end = std::min(ptcb_params.LT, ptcb_params.defect_size);

      for (int i = 0; i< Nc*Nc; ++i){ //TODO: parallelize the copying in index i
        auto subview1 = Kokkos::subview(gauge_r[1][i], 
                                        ptcb_params.LX-1,
                                        std::make_pair(0, y_end),
                                        std::make_pair(0, z_end),
                                        std::make_pair(0, t_end));
        auto subview2 = Kokkos::subview(gauge_s[1][i],
                                        ptcb_params.LX-1,
                                        std::make_pair(0, y_end),
                                        std::make_pair(0, z_end),
                                        std::make_pair(0, t_end));
      
        // Create a temporary view of the same shape (on the same memory space)
        using SubViewType = decltype(subview1);
        auto layout = subview1.layout();
        SubViewType temp("temp", layout);
        // Swap the data using deep_copy
        Kokkos::deep_copy(temp, subview1);   // Save first region in temp
        Kokkos::deep_copy(subview1, subview2); // Copy second region into first
        Kokkos::deep_copy(subview2, temp);     // Copy temp into second region
      }
    }

    T get_delta_S_swap(int r){
      // returns the change in the action caused by swapping defect areas
      // hmcSims[r] swaps with hmcSims[s = r+1] and the difference in action is calculated
      int s = (r + 1) % hmcSims.size();
      std::cout << "s: " << s << "\n";
      T S_r_r {hmcSims[r]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[r], false)};
      T S_s_s {hmcSims[s]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[s], false)};
      
      std::cout << "S_r_r: " << S_r_r << "\n";
      std::cout << "S_s_s: " << S_s_s << "\n";

      this->swap_areas(r, s);
      
      T S_r_s {hmcSims[r]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[r], false)};
      T S_s_r {hmcSims[s]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[s], false)};
      
      std::cout << "S_r_s: " << S_r_s << "\n";
      std::cout << "S_s_r: " << S_s_r << "\n";
      
      return S_r_s - S_r_r + S_s_r - S_s_s;

    }

    bool ptbc_hmc_step(){
      //step each hmc, this should later be parallelized
      std::uniform_int_distribution<int> int_dist(0, hmcSims.size());
      bool accept {true};
      bool tmp {true};
      for (int i=0; i<hmcSims.size(); ++i){
        tmp = hmcSims[i]->hmc_step();
        std::cout << "Accept[" << i << "]: " << tmp << "\n";
        if (i==0) {accept = tmp;} //since operators are measured on periodic boundary conditions, save that acceptance.
      }

      int swap_start = int(dist(mt)*(hmcSims.size()+1));
      std::cout << "swap_start: " << swap_start << "\n";
      int index {0};
      for (int i = 0; i<hmcSims.size(); ++i){
        index = (i + swap_start) % hmcSims.size();
        std::cout << "swap_index: " << index << "\n";
        T dS = this->get_delta_S_swap(index);
        std::cout << "Delta S: " << dS << "\n";
        bool swap_accept {true};
        if(dS > 0.0) {
          if(dist(mt) > Kokkos::exp(-dS)) {
            this->swap_areas(index, (index + 1) % hmcSims.size()); //swap back if not accepted 
            swap_accept = false;
          }
        }
        // TODO: add logging of swap acceptances
      }
      return accept;
    }

  }; // class PTBC
} //namespace klft
