#include "PTBC.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HamiltonianField.hpp"
#include "InputParser.hpp"
#include "SimulationLogging.hpp"
#include <mpi.h>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

std::string ranked_filename(const std::string &base_filename, int rank) {
  auto pos = base_filename.rfind('.');
  if (pos == std::string::npos) {
    // No extension → just append
    return base_filename + ".rank" + std::to_string(rank);
  } else {
    // Insert before extension
    return base_filename.substr(0, pos) + ".rank" + std::to_string(rank) +
           base_filename.substr(pos);
  }
}

int PTBC_execute(const std::string &input_file,
                 const std::string &output_directory) {

  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  // const int tuning =
  //     std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 1;
  // setTuning(tuning);
  // // if tuning is enbled, check if the user has set the
  // // KLFT_CACHE_FILE environment variable
  // if (tuning) {
  //   const char *cache_file = std::getenv("KLFT_CACHE_FILE");
  //   // if it exists, read the cache
  //   if (cache_file) {
  //     if (KLFT_VERBOSITY > 0) {
  //       printf("Reading cache file: %s\n", cache_file);
  //     }
  //     readTuneCache(cache_file);
  //   }
  // }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank %d of %d\n", rank, size);

  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  PTBCSimulationLoggingParams ptbcSimLogParams;
  Integrator_Params integratorParams;
  FermionMonomial_Params fermionParams;
  auto resParsef = parseInputFile(input_file, output_directory, fermionParams);
  GaugeMonomial_Params gaugeMonomialParams;
  bool inputFileParsedCorrectly =
      (parseInputFile(input_file, output_directory, gaugeObsParams) &&
       parseInputFile(input_file, output_directory, hmcParams) &&
       parseInputFile(input_file, output_directory, simLogParams) &&
       parseInputFile(input_file, output_directory, ptbcParams) &&
       parseInputFile(input_file, output_directory, integratorParams) &&
       abs(resParsef) &&
       parseInputFile(input_file, output_directory, gaugeMonomialParams) &&
       parseInputFile(input_file, output_directory, ptbcSimLogParams));
  if (!inputFileParsedCorrectly) {
    printf("Error parsing input file\n");
    return -1;
  }
  gaugeObsParams.wilson_flow_params.beta = gaugeMonomialParams.beta;

  simLogParams.log_filename = ranked_filename(simLogParams.log_filename, rank);

  if (ptbcParams.defects.size() != size) {
    printf("Error: Number of defects (%zu) does not match size (%d)\n",
           ptbcParams.defects.size(), size);
    return -1;
  }

  ptbcParams.gaugeObsParams = gaugeObsParams;
  ptbcParams.simLogParams = simLogParams;
  ptbcParams.ptbcSimLogParams = ptbcSimLogParams;
  ptbcParams.gauge_params = gaugeMonomialParams;

  // DEBUG_MPI_PRINT("%s", gaugeMonomialParams.to_string().c_str());
  RNGType rng(hmcParams.seed + rank);
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);

  if (hmcParams.coldStart) {
    if (hmcParams.Ndims == 4) {
      defectParams<4> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4>;
        typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              identitySUN<1>(), dParams);
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
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_4_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);

      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4>;
        typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               identitySUN<2>(), dParams);
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
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_4_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);

      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        // return 1;

        // using DGaugeFieldType =
        //     DeviceGaugeFieldType<4, 1, GaugeFieldKind::PTBC>;
        // using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4>;
        // typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
        //                                       hmcParams.L2, hmcParams.L3,
        //                                       identitySUN<1>(), dParams);
        // typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
        //                                     hmcParams.L2, hmcParams.L3,
        //                                     traceT(identitySUN<1>()));
        // typename DSpinorFieldType::type s_4_U1(hmcParams.L0, hmcParams.L1,
        //                                        hmcParams.L2, hmcParams.L3,
        //                                        0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_4_U1, a_4_U1, s_4_U1, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_4_U1, a_4_U1);
        //
        //
        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);
        // if (resParsef > 0) {
        //   auto diracParams =
        //       getDiracParams<4>(g_4_U1.dimensions, fermionParams);
        //   hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
        //                            DSpinorFieldType>(s_4_U1, diracParams,
        //                                              fermionParams.tol, rng,
        //                                              0);
        // }
        //
        // using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        //
        // run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
        //          simLogParams);
        //
      }
    } else if (hmcParams.Ndims == 3) {

      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 3D\n");
        return 1;
      }

      defectParams<3> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 1, 4>;
        typename DGaugeFieldType::type g_3_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, identitySUN<1>(),
                                              dParams);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<3>(g_3_U1.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_3_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);

      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 2, 4>;
        typename DGaugeFieldType::type g_3_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, identitySUN<2>(),
                                               dParams);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<3>(g_3_SU2.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_3_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);

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

        // const auto& dimensions = g_3_SU3.dimensions;

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams
        //         );
      }
    } else if (hmcParams.Ndims == 2) {

      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 2D\n");
        return 1;
      }

      defectParams<2> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 1, 4>;
        typename DGaugeFieldType::type g_2_U1(hmcParams.L0, hmcParams.L1,
                                              identitySUN<1>(), dParams);
        typename DAdjFieldType::type a_2_U1(hmcParams.L0, hmcParams.L1,
                                            traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_2_U1(hmcParams.L0, hmcParams.L2, 0);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<2>(g_2_U1.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_2_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);

      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 2, 4>;
        typename DGaugeFieldType::type g_2_SU2(hmcParams.L0, hmcParams.L1,
                                               identitySUN<2>(), dParams);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<2>(g_2_SU2.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_2_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);

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

        // const auto& dimensions = g_2_SU3.dimensions;

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, hamiltonian_field, integrator, rng, dist,
        // mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams
        //         );
      }
    }
  } else { // Hotstart
    if (hmcParams.Ndims == 4) {
      defectParams<4> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4>;
        typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3, rng,
                                              hmcParams.rngDelta, dParams);
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
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_4_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4>;
        typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3, rng,
                                               hmcParams.rngDelta, dParams);
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
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_4_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        if (KLFT_VERBOSITY > 1) {
          printf("Running PTBC with Nc = %zu, Ndims = %d, L0 = %d, L1 = %d, "
                 "L2 = %d, L3 = %d\n",
                 hmcParams.Nc, hmcParams.Ndims, hmcParams.L0, hmcParams.L1,
                 hmcParams.L2, hmcParams.L3);
        }

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
      }
    } else if (hmcParams.Ndims == 3) {

      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 3D\n");
        return 1;
      }
      defectParams<3> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 1, 4>;
        typename DGaugeFieldType::type g_3_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, rng,
                                              hmcParams.rngDelta, dParams);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<3>(g_3_U1.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_3_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 2, 4>;
        typename DGaugeFieldType::type g_3_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, rng,
                                               hmcParams.rngDelta, dParams);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<3>(g_3_SU2.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_3_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
      }
    } else if (hmcParams.Ndims == 2) {
      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 2D\n");
        return 1;
      }
      defectParams<2> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 1, 4>;
        typename DGaugeFieldType::type g_2_U1(hmcParams.L0, hmcParams.L1, rng,
                                              hmcParams.rngDelta, dParams);
        typename DAdjFieldType::type a_2_U1(hmcParams.L0, hmcParams.L1,
                                            traceT(identitySUN<1>()));
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<2>(g_2_U1.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_2_U1, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 2, 4>;
        typename DGaugeFieldType::type g_2_SU2(hmcParams.L0, hmcParams.L1, rng,
                                               hmcParams.rngDelta, dParams);
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
        if (resParsef > 0) {
          auto diracParams =
              getDiracParams<2>(g_2_SU2.dimensions, fermionParams);
          hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator,
                                   DSpinorFieldType>(s_2_SU2, diracParams,
                                                     fermionParams.tol, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);

        run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
                 simLogParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
      }
    }
  }
  // if tuning is enabled, write the cache file
  // if (KLFT_TUNING) {
  //   const char *cache_file = std::getenv("KLFT_CACHE_FILE");
  //   if (cache_file) {
  //     writeTuneCache(cache_file);
  //   } else {
  //     printf("KLFT_CACHE_FILE not set\n");
  //   }
  // }
  return 0;
}
} // namespace klft
