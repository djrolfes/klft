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

// this file defines input parser for running different simulations
// we are going to overload the the fuction parseInputFile for all
// different Param structs

#pragma once
#include "GaugeObservable.hpp"
#include "HMC_Params.hpp"
#include "Metropolis_Params.hpp"
#include "PTBC.hpp"
#include "SimulationLogging.hpp"
#include <yaml-cpp/yaml.h>

namespace klft {

// get MetropolisParams from input file
inline bool parseInputFile(const std::string &filename,
                           const std::string &output_directory,
                           MetropolisParams &metropolisParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["MetropolisParams"]) {
      const auto &mp = config["MetropolisParams"];
      // general parameters
      metropolisParams.Ndims = mp["Ndims"].as<index_t>(4);
      metropolisParams.L0 = mp["L0"].as<index_t>(32);
      metropolisParams.L1 = mp["L1"].as<index_t>(32);
      metropolisParams.L2 = mp["L2"].as<index_t>(32);
      metropolisParams.L3 = mp["L3"].as<index_t>(32);
      metropolisParams.nHits = mp["nHits"].as<index_t>(10);
      metropolisParams.nSweep = mp["nSweep"].as<index_t>(1000);
      metropolisParams.seed = mp["seed"].as<index_t>(1234);
      // parameters specific to the GaugeField
      metropolisParams.Nd = mp["Nd"].as<size_t>(4);
      metropolisParams.Nc = mp["Nc"].as<size_t>(2);
      // parameters specific to the Wilson action
      metropolisParams.beta = mp["beta"].as<double>(1.0);
      metropolisParams.delta = mp["delta"].as<double>(0.1);
      // ...
      // add more parameters above this as needed
    } else {
      printf("Error: MetropolisParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get GaugeObservableParams from input file
inline int parseInputFile(const std::string &filename,
                          const std::string &output_directory,
                          GaugeObservableParams &gaugeObservableParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse GaugeObservableParams
    if (config["GaugeObservableParams"]) {
      const auto &gp = config["GaugeObservableParams"];
      // interval between measurements
      gaugeObservableParams.measurement_interval =
          gp["measurement_interval"].as<size_t>(0);
      // whether to measure the topological charge
      gaugeObservableParams.measure_topological_charge =
          gp["measure_topological_charge"].as<bool>(false);
      // whether to measure the plaquette
      gaugeObservableParams.measure_plaquette =
          gp["measure_plaquette"].as<bool>(false);
      // whether to measure the temporal Wilson loop
      gaugeObservableParams.measure_wilson_loop_temporal =
          gp["measure_wilson_loop_temporal"].as<bool>(false);
      // whether to measure the mu-nu Wilson loop
      gaugeObservableParams.measure_wilson_loop_mu_nu =
          gp["measure_wilson_loop_mu_nu"].as<bool>(false);

      // pairs of (L,T) for the temporal Wilson loop
      if (gp["W_temp_L_T_pairs"]) {
        for (const auto &pair : gp["W_temp_L_T_pairs"]) {
          gaugeObservableParams.W_temp_L_T_pairs.push_back(
              IndexArray<2>({pair[0].as<index_t>(), pair[1].as<index_t>()}));
        }
      }

      // pairs of (mu,nu) for the mu-nu Wilson loop
      if (gp["W_mu_nu_pairs"]) {
        for (const auto &pair : gp["W_mu_nu_pairs"]) {
          gaugeObservableParams.W_mu_nu_pairs.push_back(
              IndexArray<2>({pair[0].as<index_t>(), pair[1].as<index_t>()}));
        }
      }

      // pairs of (Lmu,Lnu) for the Wilson loop
      if (gp["W_Lmu_Lnu_pairs"]) {
        for (const auto &pair : gp["W_Lmu_Lnu_pairs"]) {
          gaugeObservableParams.W_Lmu_Lnu_pairs.push_back(
              IndexArray<2>({pair[0].as<index_t>(), pair[1].as<index_t>()}));
        }
      }

      // filenames for the measurements
      gaugeObservableParams.topological_charge_filename =
          output_directory +
          gp["topological_charge_filename"].as<std::string>("");
      gaugeObservableParams.plaquette_filename =
          output_directory + gp["plaquette_filename"].as<std::string>("");
      gaugeObservableParams.W_temp_filename =
          output_directory + gp["W_temp_filename"].as<std::string>("");
      gaugeObservableParams.W_mu_nu_filename =
          output_directory + gp["W_mu_nu_filename"].as<std::string>("");

      // whether to write to file
      gaugeObservableParams.write_to_file = gp["write_to_file"].as<bool>(false);

      // flush interval
      gaugeObservableParams.flush = gp["flush"].as<size_t>(25);
      // ...
      // add more parameters above this line as needed
    } else {
      printf("Error: GaugeObservableParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("(GaugeObservableParams) Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get GaugeObservableParams from input file
inline bool parseInputFile(const std::string &filename,
                           const std::string &output_directory,
                           PTBCParams &ptbcParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse GaugeObservableParams
    if (config["PTBCParams"]) {
      const auto &pp = config["PTBCParams"];
      ptbcParams.defect_length = pp["defect_length"].as<index_t>(1);
      if (pp["defect_values"]) {
        auto defect_node = pp["defect_values"];
        // Set n_sims based on the number of values
        ptbcParams.n_sims = defect_node.size();
        // Resize and read values into std::vector
        ptbcParams.defects.resize(ptbcParams.n_sims);
        for (size_t i = 0; i < defect_node.size(); ++i) {
          ptbcParams.defects[i] = defect_node[i].as<real_t>();
        }
      } else {
        // Default: single defect value
        ptbcParams.n_sims = 1;
        ptbcParams.defects = {1.0}; // will this error without resizing?
      } // ...
    } else {
      printf("Error: PTBCParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("(PTBCParams) Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get HMCParams from input file
inline bool parseInputFile(const std::string &filename,
                           const std::string &output_directory,
                           HMCParams &hmcParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["HMCParams"]) {
      const auto &mp = config["HMCParams"];
      // general parameters
      hmcParams.Ndims = mp["Ndims"].as<index_t>(4);
      hmcParams.L0 = mp["L0"].as<index_t>(32);
      hmcParams.L1 = mp["L1"].as<index_t>(32);
      hmcParams.L2 = mp["L2"].as<index_t>(32);
      hmcParams.L3 = mp["L3"].as<index_t>(32);
      hmcParams.seed = mp["seed"].as<index_t>(1234);
      hmcParams.rngDelta = mp["rngDelta"].as<double>(1.0);
      hmcParams.coldStart = mp["coldStart"].as<bool>(false);
      // parameters specific to the GaugeField
      hmcParams.Nd = mp["Nd"].as<size_t>(4);
      hmcParams.Nc = mp["Nc"].as<size_t>(2);
      // add more parameters above this as needed
    } else {
      printf("Error: HMCParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("(HMC Params) Error parsing input file: %s\n", e.what());
    return false;
  }
}

inline int parseInputFile(const std::string &filename,
                          const std::string &output_directory,
                          GaugeMonomial_Params &gmparams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse HMCParams
    if (config["Gauge Monomial"]) {
      const auto &mp = config["Gauge Monomial"];
      gmparams.level = mp["level"].as<index_t>(1);
      gmparams.beta = mp["beta"].as<real_t>(1.0);
    } else {
      printf(
          "Error: You have to provide a  Gauge Monomial  in the input file\n");
      // TODO: Change back
      // return false;
    }
  } catch (const YAML::Exception &e) {
    printf("(Gauge Monomial) Error parsing input file: %s\n", e.what());
    return false;
  }
  return true;
}

inline int parseInputFile(const std::string &filename,
                          const std::string &output_directory,
                          FermionMonomial_Params &fermionParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse FermionParams
    if (config["Fermion Monomial"]) {
      const auto &fp = config["Fermion Monomial"];
      fermionParams.level = fp["level"].as<index_t>(0);
      fermionParams.fermion_type = fp["fermion"].as<std::string>("HWilson");
      fermionParams.Solver = fp["solver"].as<std::string>("CG");
      fermionParams.RepDim = fp["RepDim"].as<size_t>(4);
      fermionParams.kappa = fp["kappa"].as<real_t>(0.1);

      fermionParams.tol = fp["tol"].as<real_t>(1e-8);
    } else {
      // No Fermions
      return -1;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("(Fermion Params) Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get SimulationLoggingParams from input file
inline int parseInputFile(const std::string &filename,
                          const std::string &output_directory,
                          SimulationLoggingParams &simParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["SimulationLoggingParams"]) {
      const auto &mp = config["SimulationLoggingParams"];
      // general parameters
      simParams.log_interval = mp["log_interval"].as<size_t>(0);
      simParams.log_filename =
          output_directory + mp["log_filename"].as<std::string>("");
      simParams.write_to_file = mp["write_to_file"].as<bool>(false);
      simParams.flush = mp["flush"].as<size_t>(25);

      simParams.log_delta_H = mp["log_delta_H"].as<bool>(false);
      simParams.log_acceptance = mp["log_acceptance"].as<bool>(false);
      simParams.log_accept = mp["log_accept"].as<bool>(false);
      simParams.log_time = mp["log_time"].as<bool>(false);
    } else {
      printf("Error: SimulationLoggingParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("(SimulationLoggingParams) Error parsing input file: %s\n",
           e.what());
    return false;
  }
}

inline int parseInputFile(const std::string &filename,
                          const std::string &output_directory,
                          Integrator_Params &intParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse Integrator_Params
    if (config["Integrator"]) {
      const auto &ip = config["Integrator"];

      intParams.tau = ip["tau"] ? ip["tau"].as<real_t>() : 0.01;
      intParams.nsteps = ip["nSteps"] ? ip["nSteps"].as<index_t>() : 10;

      if (ip["Monomials"]) {
        const auto &monomials = ip["Monomials"];
        if (!monomials.IsSequence()) {
          printf("Error: 'Monomials' must be a YAML sequence\n");
          return false;
        }

        for (const auto &mon : monomials) {
          auto mono = mon.as<Integrator_Monomial_Params>();
          intParams.monomials.push_back(mono);
        }
        // Sort monomials by level (descending)
        std::sort(intParams.monomials.begin(), intParams.monomials.end(),
                  [](const Integrator_Monomial_Params &a,
                     const Integrator_Monomial_Params &b) {
                    return a.level < b.level;
                  });
      } else {
        printf("Warning: No monomials found\n");
      }

    } else {
      printf("Error: Integrator parameters not found in input file\n");
      return false;
    }

    return true;
  } catch (const YAML::Exception &e) {
    printf("YAML Error:  %s\n", e.what());
    return false;
  }
}

// get SimulationLoggingParams from input file
inline int parseInputFile(const std::string &filename,
                          const std::string &output_directory,
                          PTBCSimulationLoggingParams &simParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["PTBCSimulationLoggingParams"]) {
      const auto &mp = config["PTBCSimulationLoggingParams"];
      // general parameters
      simParams.log_interval = mp["log_interval"].as<size_t>(0);
      simParams.log_filename =
          output_directory + mp["log_filename"].as<std::string>("");
      simParams.write_to_file = mp["write_to_file"].as<bool>(false);
      simParams.flush = mp["flush"].as<size_t>(25);

      simParams.log_swap_start = mp["log_swap_start"].as<bool>(false);
      simParams.log_swap_accepts = mp["log_swap_accepts"].as<bool>(false);
      simParams.log_delta_H_swap = mp["log_delta_H_swap"].as<bool>(false);
      simParams.log_defects = mp["log_defects"].as<bool>(false);
    } else {
      printf("Error: PTBCSimulationLoggingParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception &e) {
    printf("(PTBCSimulationLoggingParams) Error parsing input file: %s\n",
           e.what());
    return false;
  }
}

inline int parseSanityChecks(const Integrator_Params &iparams,
                             const GaugeMonomial_Params &gmparams,
                             const FermionMonomial_Params &fparams,
                             const int resParsef) {
  // Check if the integrator has at least one monomial
  if (iparams.monomials.empty()) {
    printf("Error: Integrator must have at least one monomial\n");
    return false;
  }

  // Check if the gauge monomial parameters are set
  if (gmparams.beta <= 0) {
    printf("Error: Gauge Monomial beta must be positive\n");
    // TODO: Change back
    //  return false;
  }
  printf("resParsef: %d\n", resParsef);
  if (!(resParsef <= 0)) {
    if (fparams.fermion_type.empty()) {
      printf("Error: Fermion Monomial type must be specified\n");
      return false;
    }
    if (!(fparams.fermion_type == "HWilson" ||
          fparams.fermion_type == "Wilson")) {
      printf("Error: Unsupported Fermion Monomial type: %s\n",
             fparams.fermion_type.c_str());
      return false;
    }
    // Check for correct solver
    if (!(fparams.Solver == "CG" && fparams.fermion_type == "HWilson")) {
      printf("Error: Unsupported Fermion Monomial solver: %s for Fermion Type: "
             "%s\n",
             fparams.Solver.c_str(), fparams.fermion_type.c_str());
      return false;
    }

    // Check if the fermion monomial parameters are set
    if (fparams.RepDim != 4 && fparams.RepDim != 2) {
      printf("Error: Fermion Monomial RepDim must be 2 or 4\n");
      return false;
    }
    // Check if the fermion monomial parameters are set
    // if (fparams.kappa < 0) {
    //   printf("Error: Fermion Monomial kappa must be positive\n");
    //   return false;
    // }
  }
  return true;
  //
}

} // namespace klft

namespace YAML {
template <> struct convert<klft::Integrator_Monomial_Params> {
  static int decode(const Node &node, klft::Integrator_Monomial_Params &rhs) {
    if (!node["level"]) {
      printf("Monomial missing required field 'level'\n");
      return false;
    }
    rhs.type = node["Type"] ? node["Type"].as<std::string>() : "Leapfrog";
    rhs.level = node["level"] ? node["level"].as<klft::index_t>() : 0;
    rhs.steps = node["steps"] ? node["steps"].as<klft::index_t>() : 20;
    return true;
  }
};
} // namespace YAML
