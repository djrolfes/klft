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
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"
#include "updateMomentumFermion.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

// this will not work, have also give it the fields, for Fermions one has to do
// more (probably)
// Still need to add check for different Dirac Operators
template <typename DGaugeFieldType,
          typename DAdjFieldType,
          typename DSpinorFieldType>
std::shared_ptr<Integrator> createIntegrator(
    typename DGaugeFieldType::type& g_in,
    typename DAdjFieldType::type& a_in,
    typename DSpinorFieldType::type& s_in,
    const Integrator_Params& integratorParams,
    const GaugeMonomial_Params& gaugeMonomialParams,
    const FermionMonomial_Params& fermionParams,
    const int& resParsef) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  constexpr const size_t Nd = rank;
  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  // Create the integrator based on the type
  if (integratorParams.monomials.empty()) {
    printf("Error: Integrator must have at least one monomial\n");
    return nullptr;
  }
  // startingpoit of integrator chain
  std::shared_ptr<Integrator> nested_integrator = nullptr;
  for (const auto& monomial : integratorParams.monomials) {
    std::shared_ptr<Integrator> integrator = nullptr;
    if (monomial.level == 0) {
      // if the level is 0, we create a new integrator with nullptr as inner
      // integrator
      if (gaugeMonomialParams.level == 0) {
        UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
        UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType> update_p(
            g_in, a_in, gaugeMonomialParams.beta);
        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr, std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p));
        } else {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr, std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p));
        }

      } else if (fermionParams.level == 0 && resParsef > 0) {
        // if the level is 0, we create a new integrator with nullptr as inner
        // integrator
        if (fermionParams.RepDim == 4) {
          auto diracParams =
              getDiracParams<rank>(g_in.dimensions, fermionParams);
          UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
          UpdateMomentumFermion<
              DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
              HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
              CGSolver<HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                       DSpinorFieldType, DGaugeFieldType>>
              update_p(s_in, g_in, a_in, diracParams, fermionParams.tol);

          if (monomial.type == "Leapfrog") {
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                std::make_shared<UpdateMomentumFermion<
                    DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
                    HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                    CGSolver<
                        HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                        DSpinorFieldType, DGaugeFieldType>>>(update_p));

          } else {
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                std::make_shared<UpdateMomentumFermion<
                    DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
                    HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                    CGSolver<
                        HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                        DSpinorFieldType, DGaugeFieldType>>>(update_p));
          }
        } else {
          printf("Error: Fermion RepDim must be 4\n");
          return nullptr;
        }
      }
      nested_integrator = integrator;
    } else if (gaugeMonomialParams.level == monomial.level) {
      // if the level is the same, we create a new integrator with the
      // previous one as inner integrator
      UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
      UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType> update_p(
          g_in, a_in, gaugeMonomialParams.beta);
      if (monomial.type == "Leapfrog") {
        integrator = std::make_shared<LeapFrog>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p));

      } else {
        integrator = std::make_shared<LeapFrog>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p));
      }

    } else if (fermionParams.level == monomial.level && resParsef > 0) {
      // if the level is 0, we create a new integrator with nullptr as inner
      // integrator
      if (fermionParams.RepDim == 4) {
        auto diracParams = getDiracParams<rank>(g_in.dimensions, fermionParams);

        UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
        UpdateMomentumFermion<
            DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
            HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
            CGSolver<HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                     DSpinorFieldType, DGaugeFieldType>>
            update_p(s_in, g_in, a_in, diracParams, fermionParams.tol);

        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<UpdateMomentumFermion<
                  DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
                  HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                  CGSolver<
                      HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                      DSpinorFieldType, DGaugeFieldType>>>(update_p));
        } else {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<UpdateMomentumFermion<
                  DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
                  HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                  CGSolver<
                      HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                      DSpinorFieldType, DGaugeFieldType>>>(update_p));
        }
      } else {
        printf("Error: Fermion RepDim must be 4\n");
        return nullptr;
      }
    }
    nested_integrator = integrator;
  }

  return nested_integrator;
}

int HMC_execute(const std::string& input_file,
                const std::string& output_directory) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);
  // if tuning is enbled, check if the user has set the
  // KLFT_CACHE_FILE environment variable
  if (tuning) {
    const char* cache_file = std::getenv("KLFT_CACHE_FILE");
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
  if (!parseInputFile(input_file, hmcParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  Integrator_Params integratorParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, integratorParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  FermionMonomial_Params fermionParams;
  auto resParsef = parseInputFile(input_file, fermionParams);
  if (resParsef == 0) {
    printf("Error parsing input file\n");
    return -1;
  } else if (resParsef < 0) {
    printf("Info: No Fermion Monomial detected, skipping\n");
  }
  GaugeMonomial_Params gaugeMonomialParams;
  if (!parseInputFile(input_file, gaugeMonomialParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  SimulationLoggingParams simLogParams;
  if (!parseInputFile(input_file, simLogParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseSanityChecks(integratorParams, gaugeMonomialParams, fermionParams,
                         resParsef)) {
    printf("Error in sanity checks\n");
    return -1;
  }

  //   FermionParams fparams;
  //   if (!parseInputFile(input_file, fparams)) {
  //     printf("Error parsing input file\n");
  //     return -1;
  //   }
  // print the parameters
  hmcParams.print();
  integratorParams.print();
  //   gaugeObsParams.print();
  if (resParsef > 0) {
    fermionParams.print();
  }

  gaugeMonomialParams.print();
  RNGType rng(hmcParams.seed);

  // Start building the Fields
  using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
  using DAdjFieldType = DeviceAdjFieldType<4, 2>;
  using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4>;
  typename DGaugeFieldType::type g_in(hmcParams.L0, hmcParams.L1, hmcParams.L2,
                                      hmcParams.L3, rng, hmcParams.rngDelta);
  typename DAdjFieldType::type a_in(hmcParams.L0, hmcParams.L1, hmcParams.L2,
                                    hmcParams.L3, traceT(identitySUN<2>()));
  typename DSpinorFieldType::type s_in(hmcParams.L0, hmcParams.L1, hmcParams.L2,
                                       hmcParams.L3, 0);
  // typename DSpinorFieldType::type f_in(hmcParams.L0, hmcParams.L1,
  // hmcParams.L2,
  //                                      hmcParams.L3, complex_t(0.0, 0.0));

  // Buid. the Integrator
  auto testIntegrator =
      createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
          g_in, a_in, s_in, integratorParams, gaugeMonomialParams,
          fermionParams, resParsef);
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  HField hamiltonian_field = HField(g_in, a_in);

  // Warning Works only for the specific setup
  // auto casted = std::dynamic_pointer_cast<UpdateMomentumFermion<
  //     DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
  //     HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
  //     testIntegrator->update_p);
  // if (!testIntegrator) {
  //   printf("Error creating integrator\n");
  //   return -1;
  // }
  // auto& spinorField = casted->chi;
  // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  // using Update_Q = UpdatePositionGauge<4, 2>;
  // using Update_P = UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>;

  // HField hamiltonian_field = HField(g_in, a_in);
  // Update_Q update_q(g_in, a_in);
  // Update_P update_p(g_in, a_in, gaugeMonomialParams.beta);
  // // the integrate might need to be passed into the run_HMC as an
  // argument as it
  // // contains a large amount of design decisions
  // std::shared_ptr<LeapFrog> testIntegrator =
  //     std::make_shared<LeapFrog>(integratorParams.monomials[0].steps,
  //     true,
  //                                nullptr,
  //                                std::make_shared<Update_Q>(update_q),
  //                                std::make_shared<Update_P>(update_p));
  const auto& dimensions = g_in.dimensions;
  // first we check that all the parameters are correct

  // for test make stuff manually
  // now define and run the hmc
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);
  using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
  HMC hmc(integratorParams, hamiltonian_field, testIntegrator, rng, dist, mt);
  hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
  hmc.add_kinetic_monomial(0);
  if (resParsef > 0) {
    auto diracParams = getDiracParams<4>(g_in.dimensions, fermionParams);

    using DiracOperator =
        HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>;
    hmc.add_fermion_monomial<
        DiracOperator,
        CGSolver<DiracOperator, DSpinorFieldType, DGaugeFieldType>,
        DSpinorFieldType>(s_in, diracParams, fermionParams.tol, rng, 0);
    /* code */
  }

  // timer to measure the time per step
  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};
  // hmc loop
  for (size_t step = 0; step < integratorParams.nsteps; ++step) {
    timer.reset();

    // perform hmc_step
    accept = hmc.hmc_step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n", step,
             static_cast<size_t>(accept), acc_rate, time);
    }
    // measure the gauge observables
    measureGaugeObservables<4, 2>(g_in, gaugeObsParams, step);
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    // TODO:make flushAllGaugeObservables append the Observables to the
    // files
    // -> don't lose all progress when the simulation is interupted if (step
    // % 50
    // == 0) {
    //   // flush every 50 steps as well to not lose data on program
    //   interuption
    //   // TODO: this should be set by the Params
    //   flushAllGaugeObservables(gaugeObsParams);
    // }
  }
  // flush the measurements to the files
  flushAllGaugeObservables(gaugeObsParams, output_directory);
  flushSimulationLogs(simLogParams, output_directory);
  printf("Total Acceptance rate: %f, Didn't Accept %f Configs", acc_rate,
         acc_sum);
  return 0;
  // return 1;
}
}  // namespace klft
