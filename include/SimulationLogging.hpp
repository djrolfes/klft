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
#pragma once
#include <fstream>

#include "GLOBAL.hpp"

namespace klft {
// define a struct to hold parameters related to the simulation logging
struct SimulationLoggingParams {
  size_t log_interval;       // interval between logs
  std::string log_filename;  // filename for the log
  bool write_to_file;        // whether to write logs to file
  size_t flush;  // interval to flush logs to file ,0 to flush at the end of the
                 // simulation

  // define flags for the different types of logs
  bool log_delta_H;
  bool log_acceptance;
  bool log_accept;
  bool log_time;

  // define vectors to hold the logs
  std::vector<size_t> log_steps;
  std::vector<real_t> delta_H;
  std::vector<real_t> acceptance;
  std::vector<bool> accept;
  std::vector<real_t> time;

  // constructor to initialize the parameters
  SimulationLoggingParams()
      : log_interval(0),
        flush(25),
        write_to_file(false),
        log_delta_H(false),
        log_acceptance(false),
        log_accept(false),
        log_time(false) {}
};

// define a function to log simulation information
inline void addLogData(
    SimulationLoggingParams& params,
    const size_t step,
    const real_t _delta_H = 0.0,
    const real_t _acceptance = 0.0,
    const bool _accept = false,
    const real_t _time =
        0.0) {  // TODO: add overloads for different passed parameters
  if (params.log_interval == 0 || step % params.log_interval != 0 ||
      step == 0) {
    return;
  }

  if (KLFT_VERBOSITY > 1) {
    printf("Logging Simulation Data\n");
    printf("step: %zu\n", step);
  }

  if (params.log_delta_H) {
    params.delta_H.push_back(_delta_H);
    if (KLFT_VERBOSITY > 1) {
      printf("delta_H: %11.6f\n", _delta_H);
    }
  }

  if (params.log_acceptance) {
    params.acceptance.push_back(_acceptance);
    if (KLFT_VERBOSITY > 1) {
      printf("acceptance: %11.6f\n", _acceptance);
    }
  }

  if (params.log_accept) {
    params.accept.push_back(_accept);
    if (KLFT_VERBOSITY > 1) {
      printf("accept: %s\n", params.accept.back() ? "true" : "false");
    }
  }

  if (params.log_time) {
    params.time.push_back(_time);
    if (KLFT_VERBOSITY > 1) {
      printf("time: %11.6f\n", _time);
    }
  }

  params.log_steps.push_back(step);
}

inline void flushSimulationLogs(const SimulationLoggingParams& params,
                                const bool HEADER = true) {
  // check if write_to_file is enabled
  if (!params.write_to_file) {
    if (KLFT_VERBOSITY > 0) {
      printf("write_to_file is not enabled\n");
    }
    return;
  }

  // open the log file
  std::ofstream file(params.log_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open log file %s\n", params.log_filename.c_str());
    return;
  }

  // write header if required
  if (HEADER) {
    file << "# step";
    if (params.log_acceptance) {
      file << ", acceptance";
    }
    if (params.log_delta_H) {
      file << ", delta_H";
    }
    if (params.log_accept) {
      file << ", accept";
    }
    if (params.log_time) {
      file << ", time";
    }
    file << "\n";
  }

  // write the logs
  for (size_t i = 0; i < params.log_steps.size(); ++i) {
    file << params.log_steps[i];
    if (params.log_acceptance) {
      file << ", " << params.acceptance[i];
    }
    if (params.log_delta_H) {
      file << ", " << params.delta_H[i];
    }
    if (params.log_accept) {
      file << ", " << (params.accept[i] ? 1 : 0);
    }
    if (params.log_time) {
      file << ", " << params.time[i];
    }
    file << "\n";
  }

  // close the file
  file.close();
}
inline void clearSimulationLogs(SimulationLoggingParams& params) {
  // clear the logs
  params.log_steps.clear();
  params.delta_H.clear();
  params.acceptance.clear();
  params.accept.clear();
  params.time.clear();
}

}  // namespace klft