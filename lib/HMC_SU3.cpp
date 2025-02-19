#include "../include/klft.hpp"
#include "../include/HMC.hpp"
#include <iostream>
#include <fstream>

namespace klft {

  template <typename T>
  void HMC_SU3_4D(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename) {
    std::cout << "Running HMC_SU3_4D" << std::endl;
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
      using Group = SU3<T>;
      using Adjoint = AdjointSU3<T>;
      using GaugeFieldType = GaugeField<T,Group,4,3>;
      using AdjointFieldType = AdjointField<T,Adjoint,4,3>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      using HamiltonianFieldType = HamiltonianField<T,Group,Adjoint,4,3>;
      RNG rng = RNG(seed);
      std::mt19937 mt(seed);
      std::uniform_real_distribution<T> dist(0.0,1.0);
      GaugeFieldType gauge_field = GaugeFieldType(LX,LY,LZ,LT);
      gauge_field.set_random(0.5,RNG(seed*2));
      AdjointFieldType adjoint_field = AdjointFieldType(LX,LY,LZ,LT);
      HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
      HMC_Params params(n_steps,tau);
      HMC<T,Group,Adjoint,RNG,4,3> hmc(params,rng,dist,mt);
      hmc.add_kinetic_monomial(0);
      hmc.add_gauge_monomial(beta,0);
      hmc.add_hamiltonian_field(hamiltonian_field);
      hmc.set_integrator(LEAPFROG);
      T plaq = hamiltonian_field.gauge_field.get_plaquette();
      std::cout << "Starting Plaquette: " << plaq << std::endl;
      std::cout << "Starting HMC: " << std::endl;
      bool accept;
      size_t n_accept = 0;
      auto hmc_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_traj; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        accept = hmc.hmc_step();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> traj_time = end_time - start_time;
        if(accept) n_accept++;
        plaq = hamiltonian_field.gauge_field.get_plaquette();
        std::cout << "Traj: " << i << " Accept: " << accept << " Plaquette: " << plaq << " Time: " << traj_time.count() << " Acceptance Rate: " << T(n_accept)/T(i+1) << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << accept << ", " << plaq << ", " << traj_time.count() << ", " << T(n_accept)/T(i+1) << std::endl;
        }
      }
      auto hmc_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> hmc_time = hmc_end_time - hmc_start_time;
      std::cout << "HMC Time: " << hmc_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template <typename T>
  void HMC_SU3_3D(const size_t &LX, const size_t &LY, const size_t &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename) {
    std::cout << "Running HMC_SU3_3D" << std::endl;
    std::cout << "Gauge Field Dimensions:" << std::endl;
    std::cout << "LX = " << LX << std::endl;
    std::cout << "LY = " << LY << std::endl;
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
      using Group = SU3<T>;
      using Adjoint = AdjointSU3<T>;
      using GaugeFieldType = GaugeField<T,Group,3,3>;
      using AdjointFieldType = AdjointField<T,Adjoint,3,3>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      using HamiltonianFieldType = HamiltonianField<T,Group,Adjoint,3,3>;
      RNG rng = RNG(seed);
      std::mt19937 mt(seed);
      std::uniform_real_distribution<T> dist(0.0,1.0);
      GaugeFieldType gauge_field = GaugeFieldType(LX,LY,LT);
      gauge_field.set_random(0.5,RNG(seed*2));
      AdjointFieldType adjoint_field = AdjointFieldType(LX,LY,LT);
      HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
      HMC_Params params(n_steps,tau);
      HMC<T,Group,Adjoint,RNG,3,3> hmc(params,rng,dist,mt);
      hmc.add_kinetic_monomial(0);
      hmc.add_gauge_monomial(beta,0);
      hmc.add_hamiltonian_field(hamiltonian_field);
      hmc.set_integrator(LEAPFROG);
      T plaq = hamiltonian_field.gauge_field.get_plaquette();
      std::cout << "Starting Plaquette: " << plaq << std::endl;
      std::cout << "Starting HMC: " << std::endl;
      bool accept;
      size_t n_accept = 0;
      auto hmc_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_traj; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        accept = hmc.hmc_step();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> traj_time = end_time - start_time;
        if(accept) n_accept++;
        plaq = hamiltonian_field.gauge_field.get_plaquette();
        std::cout << "Traj: " << i << " Accept: " << accept << " Plaquette: " << plaq << " Time: " << traj_time.count() << " Acceptance Rate: " << T(n_accept)/T(i+1) << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << accept << ", " << plaq << ", " << traj_time.count() << ", " << T(n_accept)/T(i+1) << std::endl;
        }
      }
      auto hmc_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> hmc_time = hmc_end_time - hmc_start_time;
      std::cout << "HMC Time: " << hmc_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template <typename T>
  void HMC_SU3_2D(const size_t &LX, const size_t &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename) {
    std::cout << "Running HMC_SU3_2D" << std::endl;
    std::cout << "Gauge Field Dimensions:" << std::endl;
    std::cout << "LX = " << LX << std::endl;
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
      using Group = SU3<T>;
      using Adjoint = AdjointSU3<T>;
      using GaugeFieldType = GaugeField<T,Group,2,3>;
      using AdjointFieldType = AdjointField<T,Adjoint,2,3>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      using HamiltonianFieldType = HamiltonianField<T,Group,Adjoint,2,3>;
      RNG rng = RNG(seed);
      std::mt19937 mt(seed);
      std::uniform_real_distribution<T> dist(0.0,1.0);
      GaugeFieldType gauge_field = GaugeFieldType(LX,LT);
      gauge_field.set_random(0.5,RNG(seed*2));
      AdjointFieldType adjoint_field = AdjointFieldType(LX,LT);
      HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
      HMC_Params params(n_steps,tau);
      HMC<T,Group,Adjoint,RNG,2,3> hmc(params,rng,dist,mt);
      hmc.add_kinetic_monomial(0);
      hmc.add_gauge_monomial(beta,0);
      hmc.add_hamiltonian_field(hamiltonian_field);
      hmc.set_integrator(LEAPFROG);
      T plaq = hamiltonian_field.gauge_field.get_plaquette();
      std::cout << "Starting Plaquette: " << plaq << std::endl;
      std::cout << "Starting HMC: " << std::endl;
      bool accept;
      size_t n_accept = 0;
      auto hmc_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_traj; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        accept = hmc.hmc_step();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> traj_time = end_time - start_time;
        if(accept) n_accept++;
        plaq = hamiltonian_field.gauge_field.get_plaquette();
        std::cout << "Traj: " << i << " Accept: " << accept << " Plaquette: " << plaq << " Time: " << traj_time.count() << " Acceptance Rate: " << T(n_accept)/T(i+1) << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << accept << ", " << plaq << ", " << traj_time.count() << ", " << T(n_accept)/T(i+1) << std::endl;
        }
      }
      auto hmc_end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> hmc_time = hmc_end_time - hmc_start_time;
      std::cout << "HMC Time: " << hmc_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template void HMC_SU3_4D<float>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                                  const size_t &n_traj, const size_t &n_steps, const float &tau, const float &beta,
                                  const size_t &seed, const std::string &outfilename);

  template void HMC_SU3_4D<double>(const size_t &LX, const size_t &LY, const size_t &LZ, const size_t &LT,
                                   const size_t &n_traj, const size_t &n_steps, const double &tau, const double &beta,
                                   const size_t &seed, const std::string &outfilename);

  template void HMC_SU3_3D<float>(const size_t &LX, const size_t &LY, const size_t &LT,
                                 const size_t &n_traj, const size_t &n_steps, const float &tau, const float &beta,
                                 const size_t &seed, const std::string &outfilename);

  template void HMC_SU3_3D<double>(const size_t &LX, const size_t &LY, const size_t &LT,
                                  const size_t &n_traj, const size_t &n_steps, const double &tau, const double &beta,
                                  const size_t &seed, const std::string &outfilename);

  template void HMC_SU3_2D<float>(const size_t &LX, const size_t &LT,
                                 const size_t &n_traj, const size_t &n_steps, const float &tau, const float &beta,
                                 const size_t &seed, const std::string &outfilename);

  template void HMC_SU3_2D<double>(const size_t &LX, const size_t &LT,
                                  const size_t &n_traj, const size_t &n_steps, const double &tau, const double &beta,
                                  const size_t &seed, const std::string &outfilename);                                                                                                                                        

} // namespace klft