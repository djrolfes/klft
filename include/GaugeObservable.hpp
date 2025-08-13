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

// this file defines the necessary functions to calculate
// different gauge observables to be measured during simulations

#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>

#include "FieldTypeHelper.hpp"
#include "GaugePlaquette.hpp"
#include "WilsonLoop.hpp"

namespace klft {
// define a struct to hold parameters related to the gauge observables
struct GaugeObservableParams {
  size_t measurement_interval;       // interval between measurements
  bool measure_plaquette;            // whether to measure the plaquette
  bool measure_wilson_loop_temporal; // whether to measure the temporal Wilson
                                     // loop
  bool measure_wilson_loop_mu_nu;    // whether to measure the mu-nu Wilson loop
  std::vector<Kokkos::Array<index_t, 2>>
      W_temp_L_T_pairs; // pairs of (L,T) for the temporal Wilson loop
  std::vector<Kokkos::Array<index_t, 2>>
      W_mu_nu_pairs; // pairs of (mu,nu) for the mu-nu Wilson loop
  std::vector<Kokkos::Array<index_t, 2>>
      W_Lmu_Lnu_pairs; // pairs of (Lmu,Lnu) for the Wilson loop

  // we also define vectors to hold the measurements
  std::vector<size_t>
      measurement_steps; // steps at which measurements are taken
  std::vector<real_t> plaquette_measurements; // measurements of the plaquette
  std::vector<std::vector<Kokkos::Array<real_t, 3>>>
      W_temp_measurements; // L, T and corresponding W_temp
  std::vector<std::vector<Kokkos::Array<real_t, 5>>>
      W_mu_nu_measurements; // mu, nu, Lmu, Lnu and corresponding W_mu_nu

  // finally, some filenames where the measurements will be flushed
  std::string plaquette_filename; // filename for the plaquette measurements
  std::string
      W_temp_filename; // filename for the temporal Wilson loop measurements
  std::string
      W_mu_nu_filename; // filename for the mu-nu Wilson loop measurements

  // boolean flag to indicate if the measurements are to be flushed
  bool write_to_file;

  //
  size_t flush; // interval to flush measurements to file, 0 to flush at the
                // end of the simulation
  bool flushed; // check if the measurements were flushed at least once -> used
                // to add the header line to the file

  // constructor to initialize the parameters
  // by default nothing is measured
  GaugeObservableParams()
      : measurement_interval(0), measure_plaquette(false),
        measure_wilson_loop_temporal(false), measure_wilson_loop_mu_nu(false),
        flush(25), flushed(false) {}
};

typedef enum {
  MPI_GAUGE_OBSERVABLES_PLAQUETTE = 0,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU = 1,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL = 2,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE = 3,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE = 4
} MPI_GaugeObservableTags;

template <size_t Nd, size_t Nc>
void measureGaugeObservablesPTBC(
    const typename DeviceGaugeFieldType<Nd, Nc, GaugeFieldKind::PTBC>::type
        &g_in,
    GaugeObservableParams &params, const size_t step, const int compute_rank,
    const bool do_compute = false) {

  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  real_t Plaquette;
  std::vector<Kokkos::Array<real_t, 5>> WilsonLoop_meas;
  std::vector<Kokkos::Array<real_t, 3>> WilsonTemp_measurements;

  if (do_compute) {
    // otherwise, carry out the measurements
    if (KLFT_VERBOSITY > 1) {
      printf("Measurement of Gauge Observables\n");
      printf("step: %zu\n", step);
    }
    // measure the plaquette if requested
    if (params.measure_plaquette) {
      Plaquette = GaugePlaquette<Nd, Nc, GaugeFieldKind::PTBC>(g_in);
      if (KLFT_VERBOSITY > 1) {
        printf("plaquette: %11.6f\n", Plaquette);
      }
      MPI_Send(&Plaquette, 1, mpi_real_t(), 0, MPI_GAUGE_OBSERVABLES_PLAQUETTE,
               MPI_COMM_WORLD);
    }
    if (params.measure_wilson_loop_mu_nu) {
      if (KLFT_VERBOSITY > 1) {
        printf("Wilson loop in the mu-nu plane:\n");
        printf("mu, nu, Lmu, Lnu, W_mu_nu\n");
      }
      for (const auto &pair_mu_nu : params.W_mu_nu_pairs) {
        const index_t mu = pair_mu_nu[0];
        const index_t nu = pair_mu_nu[1];
        WilsonLoop_mu_nu<Nd, Nc, GaugeFieldKind::PTBC>(
            g_in, mu, nu, params.W_Lmu_Lnu_pairs, WilsonLoop_meas);
        if (KLFT_VERBOSITY > 1) {
          for (const auto &measure : WilsonLoop_meas) {
            printf("%d, %d, %d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                   static_cast<index_t>(measure[1]),
                   static_cast<index_t>(measure[2]),
                   static_cast<index_t>(measure[3]), measure[4]);
          }
        }
      }
      size_t WilsonLoop_meas_size = WilsonLoop_meas.size();
      MPI_Send(&WilsonLoop_meas_size, 1, mpi_size_t(), 0,
               MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE, MPI_COMM_WORLD);
      if (WilsonLoop_meas_size > 0) {
        MPI_Send(WilsonLoop_meas.data(),
                 WilsonLoop_meas_size * sizeof(Kokkos::Array<real_t, 5>),
                 MPI_BYTE, 0, MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU,
                 MPI_COMM_WORLD);
      }
    }
    if (params.measure_wilson_loop_temporal) {
      // measure the Wilson loop in the temporal direction
      if (KLFT_VERBOSITY > 1) {
        printf("temporal Wilson loop:\n");
        printf("L, T, W_temp\n");
      }
      WilsonLoop_temporal<Nd, Nc, GaugeFieldKind::PTBC>(
          g_in, params.W_temp_L_T_pairs, WilsonTemp_measurements);
      size_t WilsonTemp_measurements_size = WilsonTemp_measurements.size();
      if (KLFT_VERBOSITY > 1) {
        for (const auto &measure : WilsonTemp_measurements) {
          printf("%d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]), measure[2]);
        }
      }
      MPI_Send(&WilsonTemp_measurements_size, 1, mpi_size_t(), 0,
               MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE, MPI_COMM_WORLD);
      if (WilsonTemp_measurements_size > 0) {
        MPI_Send(WilsonTemp_measurements.data(),
                 WilsonTemp_measurements_size *
                     sizeof(Kokkos::Array<real_t, 3>),
                 MPI_BYTE, 0, MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL,
                 MPI_COMM_WORLD);
      }
    }
  }

  if (rank == 0) {
    // only the computing rank measures the observables
    params.measurement_steps.push_back(step);
    if (params.measure_plaquette) {
      // send the plaquette measurement to the compute rank
      MPI_Recv(&Plaquette, 1, mpi_real_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_PLAQUETTE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      params.plaquette_measurements.push_back(Plaquette);
    }

    if (params.measure_wilson_loop_mu_nu) {
      // MPI_Recv(&WilsonLoop_meas, 1, mpi_real_t(), compute_rank, 0,
      //  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      size_t WilsonLoop_meas_size;
      MPI_Recv(&WilsonLoop_meas_size, 1, mpi_size_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (WilsonLoop_meas_size > 0) {
        WilsonLoop_meas.resize(WilsonLoop_meas_size);
        MPI_Recv(WilsonLoop_meas.data(),
                 WilsonLoop_meas_size * sizeof(Kokkos::Array<real_t, 5>),
                 MPI_BYTE, compute_rank,
                 MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      params.W_mu_nu_measurements.push_back(WilsonLoop_meas);
    }

    if (params.measure_wilson_loop_temporal) {
      // measure the Wilson loop in the temporal direction
      if (KLFT_VERBOSITY > 1) {
        printf("temporal Wilson loop:\n");
        printf("L, T, W_temp\n");
      }
      size_t WilsonTemp_measurements_size;
      MPI_Recv(&WilsonTemp_measurements_size, 1, mpi_size_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (WilsonTemp_measurements_size > 0) {
        WilsonTemp_measurements.resize(WilsonTemp_measurements_size);
        MPI_Recv(
            WilsonTemp_measurements.data(),
            WilsonTemp_measurements_size * sizeof(Kokkos::Array<real_t, 3>),
            MPI_BYTE, compute_rank, MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      if (KLFT_VERBOSITY > 1) {
        for (const auto &measure : WilsonTemp_measurements) {
          printf("%d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]), measure[2]);
        }
      }
      params.W_temp_measurements.push_back(WilsonTemp_measurements);
    }
  }
}

// define a function to measure the gauge observables
template <size_t rank, size_t Nc>
void measureGaugeObservables(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    GaugeObservableParams &params, const size_t step) {
  // check if the step is a measurement step
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }
  // otherwise, carry out the measurements
  if (KLFT_VERBOSITY > 1) {
    printf("Measurement of Gauge Observables\n");
    printf("step: %zu\n", step);
  }
  // measure the plaquette if requested
  if (params.measure_plaquette) {
    real_t P = GaugePlaquette<rank, Nc>(g_in);
    params.plaquette_measurements.push_back(P);
    if (KLFT_VERBOSITY > 1) {
      printf("plaquette: %11.6f\n", P);
    }
  }
  // measure the Wilson loop in the temporal direction
  if (params.measure_wilson_loop_temporal) {
    if (KLFT_VERBOSITY > 1) {
      printf("temporal Wilson loop:\n");
      printf("L, T, W_temp\n");
    }
    std::vector<Kokkos::Array<real_t, 3>> temp_measurements;
    WilsonLoop_temporal<rank, Nc>(g_in, params.W_temp_L_T_pairs,
                                  temp_measurements);
    if (KLFT_VERBOSITY > 1) {
      for (const auto &measure : temp_measurements) {
        printf("%d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
               static_cast<index_t>(measure[1]), measure[2]);
      }
    }
    params.W_temp_measurements.push_back(temp_measurements);
  }
  // measure the Wilson loop in the mu-nu plane
  if (params.measure_wilson_loop_mu_nu) {
    if (KLFT_VERBOSITY > 1) {
      printf("Wilson loop in the mu-nu plane:\n");
      printf("mu, nu, Lmu, Lnu, W_mu_nu\n");
    }
    std::vector<Kokkos::Array<real_t, 5>> temp_measurements;
    for (const auto &pair_mu_nu : params.W_mu_nu_pairs) {
      const index_t mu = pair_mu_nu[0];
      const index_t nu = pair_mu_nu[1];
      WilsonLoop_mu_nu<rank, Nc>(g_in, mu, nu, params.W_Lmu_Lnu_pairs,
                                 temp_measurements);
      if (KLFT_VERBOSITY > 1) {
        for (const auto &measure : temp_measurements) {
          printf("%d, %d, %d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]),
                 static_cast<index_t>(measure[2]),
                 static_cast<index_t>(measure[3]), measure[4]);
        }
      }
    }
    params.W_mu_nu_measurements.push_back(temp_measurements);
  }
  // add the step to the measurement steps
  params.measurement_steps.push_back(step);
  return;
}

// flush the plaquette measurements to disk
inline void flushPlaquette(std::ofstream &file,
                           const GaugeObservableParams &params,
                           const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if plaquette measurements are available
  if (!params.measure_plaquette) {
    printf("Error: no plaquette measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, plaquette\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.plaquette_measurements[i] << "\n";
  }
}

// flush the temporal Wilson loop measurements to disk
inline void flushWilsonLoopTemporal(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if temporal Wilson loop measurements are available
  if (!params.measure_wilson_loop_temporal) {
    printf("Error: no temporal Wilson loop measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, L, T, W_temp\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_temp_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << "\n";
    }
  }
}

// flush the mu-nu Wilson loop measurements to disk
inline void flushWilsonLoopMuNu(std::ofstream &file,
                                const GaugeObservableParams &params,
                                const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if mu-nu Wilson loop measurements are available
  if (!params.measure_wilson_loop_mu_nu) {
    printf("Error: no mu-nu Wilson loop measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, mu, nu, Lmu, Lnu, W_mu_nu\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_mu_nu_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << ", " << measurement[3]
           << ", " << measurement[4] << "\n";
    }
  }
}

// function to clear all measurements
inline void clearAllGaugeObservables(GaugeObservableParams &params) {
  params.measurement_steps.clear();
  params.plaquette_measurements.clear();
  params.W_temp_measurements.clear();
  params.W_mu_nu_measurements.clear();
  // ...
  // add more clear functions for other observables here
}

// define a global function to flush all measurements
inline void
forceflushAllGaugeObservables(GaugeObservableParams &params,
                              const bool clear_after_flush = false,
                              const int &p = std::cout.precision()) {
  auto _ = std::setprecision(p);
  // check if write_to_file is enabled
  if (!params.write_to_file) {
    printf("write_to_file is not enabled\n");
    return;
  }
  bool HEADER = !params.flushed; // write header only once
  // flush plaquette measurements
  if (params.measure_plaquette && params.plaquette_filename != "") {
    std::ofstream file(params.plaquette_filename, std::ios::app);
    flushPlaquette(file, params, HEADER);
    file.close();
  }
  // flush temporal Wilson loop measurements
  if (params.measure_wilson_loop_temporal && params.W_temp_filename != "") {
    std::ofstream file(params.W_temp_filename, std::ios::app);
    flushWilsonLoopTemporal(file, params, HEADER);
    file.close();
  }
  // flush mu-nu Wilson loop measurements
  if (params.measure_wilson_loop_mu_nu && params.W_mu_nu_filename != "") {
    std::ofstream file(params.W_mu_nu_filename, std::ios::app);
    flushWilsonLoopMuNu(file, params, HEADER);
    file.close();
  }
  // ...
  // add more flush functions for other observables here
  if (clear_after_flush) {
    clearAllGaugeObservables(params);
  }
  params.flushed = true; // set flushed to true after flushing
}

// check if the current step should be flushed, if so call
// forceflushAllGaugeObservables
inline void flushAllGaugeObservables(GaugeObservableParams &params,
                                     const size_t step,
                                     const bool clear_after_flush = false,
                                     const int &p = std::cout.precision()) {
  if (params.flush != 0 && step % params.flush == 0) {
    forceflushAllGaugeObservables(params, clear_after_flush, p);
  }
}

} // namespace klft
