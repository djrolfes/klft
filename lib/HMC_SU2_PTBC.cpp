#include "../include/klft.hpp"
#include "../include/HMC.hpp"
#include "../include/PTBC.hpp"
#include "../include/PTBCDefect.hpp"
#include "../include/PTBC_Params.hpp"
#include <iostream>
#include <fstream>

namespace klft {

  template <typename T> //This is mainly a test file that tests the compilation of the ptbc algorithm.
  void HMC_SU2_4D_PTBC(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename, const size_t &defect_length, const T &gauge_depression) {
    std::cout << "Running HMC_SU2_4D" << std::endl;
    std::cout << "Gauge Field Dimensions:" << std::endl;
    std::cout << "LX = " << LX << std::endl;
    std::cout << "LY = " << LY << std::endl;
    std::cout << "LZ = " << LZ << std::endl;
    std::cout << "LT = " << LT << std::endl;
    std::cout << "HMC Parameters:" << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "n_traj = " << n_traj << std::endl;
    std::cout << "tau = " << tau << std::endl;
    std::cout << "n_steps = " << n_steps << std::endl;
    std::cout << "seed = " << seed << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "traj, accept, plaquette, time, acceptance rate" << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = SU2<T>;
      using Adjoint = AdjointSU2<T>;
      using GaugeFieldType = GaugeField<T,Group,4,2>;
      using AdjointFieldType = AdjointField<T,Adjoint,4,2>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      using HamiltonianFieldType = HamiltonianField<T,Group,Adjoint,4,2>;
      using Defect = PTBCDefect<T, 4>;
      using PTBC_PARAMS = PTBC_Params<T, 4>;
      RNG rng = RNG(seed);
      std::mt19937 mt(seed);
      std::uniform_real_distribution<T> dist(0.0,1.0);
      //GaugeFieldType gauge_field = GaugeFieldType(LX,LY,LZ,LT);
      //gauge_field.set_random(0.5,RNG(seed*2));
      //AdjointFieldType adjoint_field = AdjointFieldType(LX,LY,LZ,LT);
      //HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
      HMC_Params hmc_params(n_steps,tau);
      PTBC_PARAMS ptbc_params(3, defect_length, LX, LY, LZ, LT); //TODO: make N_simulations an input parameterb

      //HMC<T,Group,Adjoint,RNG,4,2> hmc(hmc_params,rng,dist,mt);
      Defect defect(gauge_depression, defect_length, LX);
      //hmc.add_kinetic_monomial(0);
      //hmc.add_gauge_monomial(beta,0, defect);
      //hmc.add_hamiltonian_field(hamiltonian_field);
      //
      PTBC<T, Group, Adjoint, RNG, 4, 2> ptbc(ptbc_params, hmc_params, rng, dist, mt);
      ptbc.add_kinetic_monomials(0);
      ptbc.add_gauge_monomials(beta, 0);
      ptbc.set_integrators(LEAPFROG);


      //T plaq = hamiltonian_field.gauge_field.get_plaquette();
      //std::cout << "Starting Plaquette: " << plaq << std::endl;
      //std::cout << "Starting HMC: " << std::endl;
      bool accept;
      size_t n_accept = 0;
      auto hmc_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_traj; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        accept = ptbc.ptbc_hmc_step();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> traj_time = end_time - start_time;
        if(accept) n_accept++;
        //plaq = hamiltonian_field.gauge_field.get_plaquette();
        //std::cout << "Traj: " << i << " Accept: " << accept << " Plaquette: " << plaq << " Time: " << traj_time.count() << " Acceptance Rate: " << T(n_accept)/T(i+1) << std::endl;
        //if(outfilename != "") {
        //  outfile << i << ", " << accept << ", " << plaq << ", " << traj_time.count() << ", " << T(n_accept)/T(i+1) << std::endl;
        //}
      }
      auto hmc_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> hmc_time = hmc_end_time - hmc_start_time;
      std::cout << "HMC Time: " << hmc_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  

  template void HMC_SU2_4D_PTBC<float>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                                  const size_t &n_traj, const size_t &n_steps, const float &tau, const float &beta,
                                  const size_t &seed, const std::string &outfilename, const size_t &defect_length, const float &gauge_depression);

  template void HMC_SU2_4D_PTBC<double>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                                   const size_t &n_traj, const size_t &n_steps, const double &tau, const double &beta,
                                   const size_t &seed, const std::string &outfilename, const size_t &defect_length, const double &gauge_depression);                                                                                                                          

} // namespace klft