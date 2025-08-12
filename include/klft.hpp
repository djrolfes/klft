#pragma once
#include <string>

namespace klft {

int Metropolis(const std::string &input_file,
               const std::string &output_directory);

int HMC_execute(const std::string &input_file,
                const std::string &output_directory);

int PTBC_execute(const std::string &input_file,
                 const std::string &output_directory);
} // namespace klft
