//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/
#include "HMC_exec.hpp"

#include "../include/HMC.hpp"
#include "../include/InputParser.hpp"
#include "../include/klft.hpp"
#include "AdjointSUN.hpp"
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GaugeObservable.hpp"
#include "HMC_Params.hpp"
#include "Integrator.hpp"
#include "SimulationLogging.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int build_and_run_HMC(const std::string &input_file,
                      const std::string &output_directory) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 1;
  setTuning(tuning);
  // if tuning is enbled, check if the user has set the
  // KLFT_CACHE_FILE environment variable
  if (tuning) {
    const char *cache_file = std::getenv("KLFT_CACHE_FILE");
    // if it exists, read the cache
    if (cache_file) {
      if (KLFT_VERBOSITY > 0) {
        printf("Reading cache file: %s\n", cache_file);
      }
      readTuneCache(cache_file);
    }
  }
  HMCParams hmcParams;
  // parse the input file for HMC parameters
  if (!parseInputFile(input_file, output_directory, hmcParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  Integrator_Params integratorParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, output_directory, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, output_directory, integratorParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  FermionMonomial_Params fermionParams;
  auto resParsef = parseInputFile(input_file, output_directory, fermionParams);
  if (resParsef == 0) {
    printf("Error parsing input file\n");
    return -1;
  } else if (resParsef < 0) {
    printf("Info: No Fermion Monomial detected, skipping\n");
  }
  GaugeMonomial_Params gaugeMonomialParams;
  if (!parseInputFile(input_file, output_directory, gaugeMonomialParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  SimulationLoggingParams simLogParams;
  if (!parseInputFile(input_file, output_directory, simLogParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseSanityChecks(integratorParams, gaugeMonomialParams, fermionParams,
                         resParsef)) {
    printf("Error in sanity checks\n");
    return -1;
  }

  // print the parameters
  hmcParams.print();
  integratorParams.print();
  //   gaugeObsParams.print();
  if (resParsef > 0) {
    fermionParams.print();
  }

  gaugeMonomialParams.print();
  RNGType rng(hmcParams.seed);
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);
  // Start building the Fields

  if (hmcParams.coldStart) {
    if (hmcParams.Ndims == 4) {
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 1>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4>;
        typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              identitySUN<1>());
        typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
                                            hmcParams.L2, hmcParams.L3,
                                            traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_4_U1(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_4_U1, a_4_U1, s_4_U1, integratorParams, gaugeMonomialParams,
                fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_4_U1, a_4_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<4>(g_4_U1.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator,
                                   DSpinorFieldType>(s_4_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4>;
        typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               identitySUN<2>());
        typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                             hmcParams.L2, hmcParams.L3,
                                             traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_4_SU2(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_4_SU2, a_4_SU2, s_4_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<4>(g_4_SU2.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator,
                                   DSpinorFieldType>(s_4_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 3) {
        // printf("Error: SU(3) isn't supported yet");
        // return 1;

        using DGaugeFieldType = DeviceGaugeFieldType<4, 3>;
        using DAdjFieldType = DeviceAdjFieldType<4, 3>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 3, 4>;
        typename DGaugeFieldType::type g_4_SU3(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               identitySUN<3>());
        typename DAdjFieldType::type a_4_SU3(hmcParams.L0, hmcParams.L1,
                                             hmcParams.L2, hmcParams.L3,
                                             traceT(identitySUN<3>()));
        typename DSpinorFieldType::type s_4_SU3(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_4_SU3, a_4_SU3, s_4_SU3, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_4_SU3, a_4_SU3);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<4>(g_4_SU3.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator,
                                   DSpinorFieldType>(s_4_SU3, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      }
    } else if (hmcParams.Ndims == 3) {
      if (hmcParams.Nc == 1) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 3D\n");
          return 1;
        }

        using DGaugeFieldType = DeviceGaugeFieldType<3, 1>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 1, 4>;

        typename DGaugeFieldType::type g_3_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, identitySUN<1>());
        typename DAdjFieldType::type a_3_U1(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<1>()));

        typename DSpinorFieldType::type s_3_U1(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_U1, a_3_U1, s_3_U1, integratorParams, gaugeMonomialParams,
                fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_U1, a_3_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 2) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 3D\n");
          return 1;
        }
        using DGaugeFieldType = DeviceGaugeFieldType<3, 2>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 2, 4>;
        typename DGaugeFieldType::type g_3_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, identitySUN<2>());
        typename DAdjFieldType::type a_3_SU2(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_3_SU2(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_SU2, a_3_SU2, s_3_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_SU2, a_3_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;

        // if (resParsef > 0) {
        //   printf("Error: Fermions are currently not supported in 3D\n");
        //   return 1;
        // }
        // using DGaugeFieldType = DeviceGaugeFieldType<3, 3>;
        // using DAdjFieldType = DeviceAdjFieldType<3, 3>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 3, 4>;
        // typename DGaugeFieldType::type g_3_SU3(hmcParams.L0, hmcParams.L1,
        //                                        hmcParams.L2,
        //                                        identitySUN<3>());
        // typename DAdjFieldType::type a_3_SU3(
        //     hmcParams.L0, hmcParams.L1, hmcParams.L2,
        //     traceT(identitySUN<3>()));
        // typename DSpinorFieldType::type s_3_SU3(hmcParams.L0, hmcParams.L1,
        //                                         hmcParams.L2, 1, 0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_3_SU3, a_3_SU3, s_3_SU3, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_3_SU3, a_3_SU3);

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      }
    } else if (hmcParams.Ndims == 2) {
      if (hmcParams.Nc == 1) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 2D\n");
          return 1;
        }

        using DGaugeFieldType = DeviceGaugeFieldType<2, 1>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 1, 4>;

        typename DGaugeFieldType::type g_2_U1(hmcParams.L0, hmcParams.L1,
                                              identitySUN<1>());
        typename DAdjFieldType::type a_2_U1(hmcParams.L0, hmcParams.L1,
                                            traceT(identitySUN<1>()));
        // Leave 4D Spinorfield such that the createIntegrator stil works
        typename DSpinorFieldType::type s_2_U1(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_U1, a_2_U1, s_2_U1, integratorParams, gaugeMonomialParams,
                fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_U1, a_2_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 2) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 3D\n");
          return 1;
        }
        using DGaugeFieldType = DeviceGaugeFieldType<2, 2>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 2, 4>;
        typename DGaugeFieldType::type g_2_SU2(hmcParams.L0, hmcParams.L1,
                                               identitySUN<2>());
        typename DAdjFieldType::type a_2_SU2(hmcParams.L0, hmcParams.L1,
                                             traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_2_SU2(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_SU2, a_2_SU2, s_2_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_SU2, a_2_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;

        // if (resParsef > 0) {
        //   printf("Error: Fermions are currently not supported in 3D\n");
        //   return 1;
        // }
        // using DGaugeFieldType = DeviceGaugeFieldType<2, 3>;
        // using DAdjFieldType = DeviceAdjFieldType<2, 3>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 3, 4>;
        // typename DGaugeFieldType::type g_2_SU3(hmcParams.L0, hmcParams.L1,
        //                                        identitySUN<3>());
        // typename DAdjFieldType::type a_2_SU3(hmcParams.L0, hmcParams.L1,
        //                                      traceT(identitySUN<3>()));
        // typename DSpinorFieldType::type s_2_SU3(hmcParams.L0, hmcParams.L1,
        // 1,
        //                                         1, 0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_2_SU3, a_2_SU3, s_2_SU3, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_2_SU3, a_2_SU3);

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      }
    }
  } else { // Hotstart
    if (hmcParams.Ndims == 4) {
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 1>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4>;
        typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3, rng,
                                              hmcParams.rngDelta);
        typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
                                            hmcParams.L2, hmcParams.L3,
                                            traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_4_U1(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_4_U1, a_4_U1, s_4_U1, integratorParams, gaugeMonomialParams,
                fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_4_U1, a_4_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<4>(g_4_U1.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator,
                                   DSpinorFieldType>(s_4_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4>;
        typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3, rng,
                                               hmcParams.rngDelta);
        typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                             hmcParams.L2, hmcParams.L3,
                                             traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_4_SU2(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_4_SU2, a_4_SU2, s_4_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<4>(g_4_SU2.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator,
                                   DSpinorFieldType>(s_4_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 3) {
        // printf("Error: SU(3) isn't supported yet");
        // return 1;

        using DGaugeFieldType = DeviceGaugeFieldType<4, 3>;
        using DAdjFieldType = DeviceAdjFieldType<4, 3>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 3, 4>;
        typename DGaugeFieldType::type g_4_SU3(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3, rng,
                                               hmcParams.rngDelta);
        typename DAdjFieldType::type a_4_SU3(hmcParams.L0, hmcParams.L1,
                                             hmcParams.L2, hmcParams.L3,
                                             traceT(identitySUN<3>()));
        typename DSpinorFieldType::type s_4_SU3(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_4_SU3, a_4_SU3, s_4_SU3, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_4_SU3, a_4_SU3);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<4>(g_4_SU3.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator,
                                   DSpinorFieldType>(s_4_SU3, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      }
    } else if (hmcParams.Ndims == 3) {
      if (hmcParams.Nc == 1) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 3D\n");
          return 1;
        }

        using DGaugeFieldType = DeviceGaugeFieldType<3, 1>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 1, 4>;

        typename DGaugeFieldType::type g_3_U1(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, rng, hmcParams.rngDelta);
        typename DAdjFieldType::type a_3_U1(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<1>()));
        // Leave 4D Spinorfield such that the createIntegrator stil works
        typename DSpinorFieldType::type s_3_U1(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_U1, a_3_U1, s_3_U1, integratorParams, gaugeMonomialParams,
                fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_U1, a_3_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 2) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 3D\n");
          return 1;
        }
        using DGaugeFieldType = DeviceGaugeFieldType<3, 2>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 2, 4>;
        typename DGaugeFieldType::type g_3_SU2(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, rng, hmcParams.rngDelta);
        typename DAdjFieldType::type a_3_SU2(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_3_SU2(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_SU2, a_3_SU2, s_3_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_SU2, a_3_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;

        // if (resParsef > 0) {
        //   printf("Error: Fermions are currently not supported in 3D\n");
        //   return 1;
        // }
        // using DGaugeFieldType = DeviceGaugeFieldType<3, 3>;
        // using DAdjFieldType = DeviceAdjFieldType<3, 3>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 3, 4>;
        // typename DGaugeFieldType::type g_3_SU3(
        //     hmcParams.L0, hmcParams.L1, hmcParams.L2, rng,
        //     hmcParams.rngDelta);
        // typename DAdjFieldType::type a_3_SU3(
        //     hmcParams.L0, hmcParams.L1, hmcParams.L2,
        //     traceT(identitySUN<3>()));
        // typename DSpinorFieldType::type s_3_SU3(hmcParams.L0, hmcParams.L1,
        //                                         hmcParams.L2, 1, 0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_3_SU3, a_3_SU3, s_3_SU3, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_3_SU3, a_3_SU3);

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      }
    } else if (hmcParams.Ndims == 2) {
      if (hmcParams.Nc == 1) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 2D\n");
          return 1;
        }

        using DGaugeFieldType = DeviceGaugeFieldType<2, 1>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 1, 4>;

        typename DGaugeFieldType::type g_2_U1(hmcParams.L0, hmcParams.L1, rng,
                                              hmcParams.rngDelta);
        typename DAdjFieldType::type a_2_U1(hmcParams.L0, hmcParams.L1,
                                            traceT(identitySUN<1>()));
        // Leave 4D Spinorfield such that the createIntegrator stil works
        typename DSpinorFieldType::type s_2_U1(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_U1, a_2_U1, s_2_U1, integratorParams, gaugeMonomialParams,
                fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_U1, a_2_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 2) {
        if (resParsef > 0) {
          printf("Error: Fermions are currently not supported in 2D\n");
          return 1;
        }
        using DGaugeFieldType = DeviceGaugeFieldType<2, 2>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 2, 4>;
        typename DGaugeFieldType::type g_2_SU2(hmcParams.L0, hmcParams.L1, rng,
                                               hmcParams.rngDelta);
        typename DAdjFieldType::type a_2_SU2(hmcParams.L0, hmcParams.L1,
                                             traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_2_SU2(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_SU2, a_2_SU2, s_2_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_SU2, a_2_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);

        run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      } else if (hmcParams.Nc == 3) {
        // printf("Error: SU(3) isn't supported yet");
        // return 1;

        // if (resParsef > 0) {
        //   printf("Error: Fermions are currently not supported in 2D\n");
        //   return 1;
        // }
        // using DGaugeFieldType = DeviceGaugeFieldType<2, 3>;
        // using DAdjFieldType = DeviceAdjFieldType<2, 3>;
        // using DSpinorFieldType = DeviceSpinorFieldType<2, 3, 4>;
        // typename DGaugeFieldType::type g_2_SU3(hmcParams.L0, hmcParams.L1,
        // rng,
        //                                        hmcParams.rngDelta);
        // typename DAdjFieldType::type a_2_SU3(hmcParams.L0, hmcParams.L1,
        //                                      traceT(identitySUN<3>()));
        // typename DSpinorFieldType::type s_2_SU3(hmcParams.L0, hmcParams.L1,
        // 1,
        //                                         1, 0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_2_SU3, a_2_SU3, s_2_SU3, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_2_SU3, a_2_SU3);

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams);
      }
    }
  }

  return 0;
  // return 1;
}
} // namespace klft
