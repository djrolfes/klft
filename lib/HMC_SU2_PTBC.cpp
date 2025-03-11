#include "../include/klft.hpp"
#include "../include/HMC.hpp"
#include "../include/PTBC.hpp"
#include "../include/PTBCDefect.hpp"
#include "../include/PTBC_Params.hpp"
#include "writeToFile.cpp"
#include <iostream>
#include <fstream>

namespace klft {

  template <typename T> //This is mainly a test file that tests the compilation of the ptbc algorithm.
  void HMC_SU2_4D_PTBC(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename, const size_t &defect_length, const size_t &n_sims) {
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
    // Write header line
    outfile << "traj, accept, plaquette, time, acceptance rate, [hmc acceptances], swap start, [swap acceptances], [delta S swap]" << std::endl;
    // Write settings as a JSON dictionary (one line)
    outfile << "{"
            << "\"LX\": " << LX << ", "
            << "\"LY\": " << LY << ", "
            << "\"LZ\": " << LZ << ", "
            << "\"LT\": " << LT << ", "
            << "\"beta\": " << beta << ", "
            << "\"n_traj\": " << n_traj << ", "
            << "\"tau\": " << tau << ", "
            << "\"n_steps\": " << n_steps << ", "
            << "\"seed\": " << seed << ", "
            << "\"defect_length\": " << defect_length << ", "
            << "\"n_sims\": " << n_sims
            << "}" << std::endl;
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
      HMC_Params hmc_params(n_steps,tau);
      PTBC_PARAMS ptbc_params(n_sims, defect_length, seed, LX, LY, LZ, LT);

      PTBC<T, Group, Adjoint, RNG, 4, 2> ptbc(ptbc_params, hmc_params, rng, dist, mt);
      ptbc.add_kinetic_monomials(0);
      ptbc.add_gauge_monomials(beta, 0);
      ptbc.set_integrators(LEAPFROG);


      T plaq = ptbc.hamiltonian_fields[0]->gauge_field.get_plaquette();
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
        plaq = ptbc.hamiltonian_fields[0]->gauge_field.get_plaquette();
        std::cout << "Traj: " << i << " Accept: " << accept << " Plaquette: " << plaq << " Time: " << traj_time.count() << " Acceptance Rate: " << T(n_accept)/T(i+1) << std::endl;
        if(outfilename != "") {
          std::string logLine = generateLogString(ptbc.ptbc_logs[0], i, plaq, static_cast<double>(n_accept)/static_cast<double>(i+1), traj_time.count()); // TODO: save plaqs, traj_times to flush the log
          outfile << logLine << std::endl;
          std::cout << "traj, accept, plaquette, time, acceptance rate, [hmc acceptances], swap start, [swap acceptances], [delta S swap]" << std::endl;
          std::cout << logLine << std::endl;
          ptbc.ptbc_logs.clear();
        }
      }
      auto hmc_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> hmc_time = hmc_end_time - hmc_start_time;
      std::cout << "HMC Time: " << hmc_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  

  template void klft::HMC_SU2_4D_PTBC<float>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                                            const size_t &n_traj, const size_t &n_steps, const float &tau, const float &beta,
                                            const size_t &seed, const std::string &outfilename, const size_t &defect_length, const size_t &n_sims);
  template void klft::HMC_SU2_4D_PTBC<double>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                                             const size_t &n_traj, const size_t &n_steps, const double &tau, const double &beta,
                                             const size_t &seed, const std::string &outfilename, const size_t &defect_length, const size_t &n_sims);
                                                                                                                         

} // namespace klft