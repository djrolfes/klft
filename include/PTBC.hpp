#pragma once
#include "klft.hpp"
#include "HMC.hpp"
#include "PTBCDefect.hpp"
#include "PTBC_Params.hpp"
#include <iostream>


namespace klft {

  struct PTBCStepLog {
      // Stores the acceptance of each HMC trajectory
      std::vector<bool> hmc_acceptances;
      // Stores the starting index used for swapping
      int swap_start_index;
      // For each swap, store whether the swap was accepted (true) or rejected (false)
      std::vector<bool> swap_acceptances;
      // Optionally, store the delta S for each swap for further analysis
      std::vector<double> delta_S_values;
      // Store c(r) values in order
      std::vector<double> cr;
    };

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
    T beta;
    std::mt19937 mt;
    std::uniform_real_distribution<T> dist;
    std::vector<std::unique_ptr<HMC<T, Group, Adjoint, RNG, Ndim, Nc>>> hmcSims;
    std::vector<std::unique_ptr<HamiltonianFieldType>> hamiltonian_fields;
    std::vector<std::unique_ptr<PTBCDefect<T, Ndim>>> defects;
    std::vector<PTBCStepLog> ptbc_logs;


    


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
      if (ptcb_params.N_simulations==1){return T(1.0);}
      return T(1.0) - T(i)/T(ptcb_params.N_simulations-1);
    }

    void init_defects(){
      for (int i = 0; i< ptcb_params.N_simulations; i++){
        defects.emplace_back(std::make_unique<PTBCDefect<T, Ndim>>(get_gauge_depression(i), ptcb_params.defect_size, ptcb_params.LX));
        //Kokkos::printf("defect[%d]: %f; defect(7,0,0,0,0): %f\n", i, defects[i]->gauge_depression, (*defects[i])(ptcb_params.LX-1,0,0,0,0));
      }
    }

    void add_hamiltonian_fields(){
      for (int i = 0; i<ptcb_params.N_simulations; i++){
        GaugeFieldType gauge_field = GaugeFieldType(ptcb_params.get_lattice_dims());
        gauge_field.set_random(0.5,RNG(ptcb_params.seed*(2+i)));//TODO: why was the seed used here and what should I use?
        AdjointFieldType adjoint_field = AdjointFieldType(ptcb_params.get_lattice_dims());
        hamiltonian_fields.emplace_back(std::make_unique<HamiltonianFieldType>(gauge_field,adjoint_field));
        hmcSims[i]->add_hamiltonian_field(*hamiltonian_fields[i]);
      }
    }

    void add_gauge_monomials(const T _beta, const unsigned int _time_scale) {
      this->beta = _beta;
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
  using complex_t = Kokkos::complex<T>;
  using DeviceView = Kokkos::View<complex_t****>;

  auto& gauge_r = hmcSims[r]->hamiltonian_field.gauge_field.gauge;
  auto& gauge_s = hmcSims[s]->hamiltonian_field.gauge_field.gauge;

  hmcSims[r]->set_defect(*defects[s]);
  hmcSims[s]->set_defect(*defects[r]);
  std::swap(this->defects[r], this->defects[s]);

  // Compute region sizes. Note: using a fixed x index.
  int y_end = std::min(ptcb_params.LY, ptcb_params.defect_size) - 1;
  int z_end = std::min(ptcb_params.LZ, ptcb_params.defect_size) - 1;
  int t_end = std::min(ptcb_params.LT, ptcb_params.defect_size) - 1;
  // For each matrix element in the link (for a given mu, here we use mu=0 as an example)
  for (int i = 0; i < Nc * Nc; ++i) {
    // Create subviews for a slice at x = LX-1 and y in [0, y_end), etc.
    auto subview1 = Kokkos::subview(gauge_r[0][i], ptcb_params.LX - 1,
                                    std::make_pair(0, y_end),
                                    std::make_pair(0, z_end),
                                    std::make_pair(0, t_end));
    auto subview2 = Kokkos::subview(gauge_s[0][i], ptcb_params.LX - 1,
                                    std::make_pair(0, y_end),
                                    std::make_pair(0, z_end),
                                    std::make_pair(0, t_end));
    
    // Create a temporary view with the same layout.
    using SubViewType = decltype(subview1);
    auto layout = subview1.layout();
    SubViewType temp("temp", layout);
    
    // Swap the data.
    Kokkos::deep_copy(temp, subview1);
    Kokkos::deep_copy(subview1, subview2);
    Kokkos::deep_copy(subview2, temp);
    
    Kokkos::fence();
  }
}


    T get_delta_S_swap(int r){
      // returns the change in the action caused by swapping defect areas
      // hmcSims[r] swaps with hmcSims[s = r+1] and the difference in action is calculated
      int s = (r + 1) % hmcSims.size();
      T S_r_r {-this->beta/3 * hmcSims[r]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[r],false)};
      T S_s_s {-this->beta/3 * hmcSims[s]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[s],false)};
      
      hmcSims[r]->set_defect(*defects[s]);
      hmcSims[s]->set_defect(*defects[r]);
      std::swap(this->defects[r], this->defects[s]);
      //std::cout << "S_r_r: " << S_r_r << "\n";
      //std::cout << "S_s_s: " << S_s_s << "\n";

      //this->swap_areas(r, s);
      
      T S_r_s {-this->beta/3 * hmcSims[r]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[r], false)};
      T S_s_r {-this->beta/3 * hmcSims[s]->hamiltonian_field.gauge_field.get_plaquette_around_defect(*defects[s], false)};
      
      //std::cout << "S_r_s: " << S_r_s << "\n";
      //std::cout << "S_s_r: " << S_s_r << "\n";
      
      return S_r_s - S_r_r + S_s_r - S_s_s;

    }

    bool ptbc_hmc_step(){
    PTBCStepLog log;
    // Step through each HMC and record the acceptance flag
    for (int i = 0; i < hmcSims.size(); ++i) {
      bool accept = hmcSims[i]->hmc_step();
      log.hmc_acceptances.push_back(accept);
    }

    // Choose a random starting index for swaps
    int swap_start = int(dist(mt) * (hmcSims.size() + 1));
    log.swap_start_index = swap_start;

    // Execute swaps and record their acceptance
    for (int i = 0; i < hmcSims.size(); ++i) {
      int index = (i + swap_start) % hmcSims.size();
      double dS = this->get_delta_S_swap(index);
      log.delta_S_values.push_back(dS);

      bool swap_accept = true;
      if(dS > 0.0) {
        if(dist(mt) > Kokkos::exp(-dS)) {
          int r = index;
          int s = (index + 1) % hmcSims.size();
          hmcSims[r]->set_defect(*defects[s]);
          hmcSims[s]->set_defect(*defects[r]);
          std::swap(this->defects[r], this->defects[s]);
          //this->swap_areas(index, (index + 1) % hmcSims.size()); // swap back if not accepted 
          swap_accept = false;
        }
      }
      log.swap_acceptances.push_back(swap_accept);
      log.cr.push_back(defects[index]->gauge_depression);
    }
    
    // Save the log for this step
    ptbc_logs.push_back(log);
    
    // Return the HMC acceptance from the first simulation as the overall acceptance (as before)
    return log.hmc_acceptances.empty() ? true : log.hmc_acceptances[0];
  }


  }; // class PTBC
} //namespace klft
