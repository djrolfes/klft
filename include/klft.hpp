#pragma once
#include <string>

namespace klft {

int Metropolis(const std::string& input_file,
               const std::string& output_directory);

int build_and_run_HMC(const std::string& input_file,
                      const std::string& output_directory);
}  // namespace klft
