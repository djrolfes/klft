#include "../include/PTBC.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>



std::string generateLogString(const klft::PTBCStepLog &log,
                              int traj,          // trajectory number
                              double plaq,       // plaquette value
                              double acceptance, // acceptance for full run
                              double swap_acceptance, // swap acceptance for full run (r = 1)
                              double traj_time)  // trajectory time
{
    std::ostringstream oss;

    // Use the first HMC acceptance as the general acceptance.
    bool general_accept = !log.hmc_acceptances.empty() ? log.hmc_acceptances[0] : false;

    // Compute the acceptance rate as the fraction of accepted HMC steps.
    double acceptance_rate = 0.0;
    if (!log.hmc_acceptances.empty()) {
        int count = 0;
        for (bool a : log.hmc_acceptances) {
            if (a) count++;
        }
        acceptance_rate = static_cast<double>(count) / log.hmc_acceptances.size();
    }

    // Build the output string.
    // Format: traj, general_accept, plaq, traj_time, acceptance_rate, topoCharge, [hmc_acceptances], swap_start_index, swap_acceptance, [swap_acceptances], [c(r) values], [delta_S_values]
    oss << traj << ", " << general_accept << ", " << plaq << ", " << traj_time << ", " << acceptance << ", " << log.topo << ", ";

    // Format hmc_acceptances list as "[val1; val2; ...]"
    oss << "[";
    for (size_t i = 0; i < log.hmc_acceptances.size(); ++i) {
        oss << log.hmc_acceptances[i];
        if (i != log.hmc_acceptances.size() - 1)
            oss << "; ";
    }
    oss << "], ";

    // Swap start index.
    oss << log.swap_start_index << ", " << swap_acceptance << ", ";

    // Format swap_acceptances list.
    oss << "[";
    for (size_t i = 0; i < log.swap_acceptances.size(); ++i) {
        oss << log.swap_acceptances[i];
        if (i != log.swap_acceptances.size() - 1)
            oss << "; ";
    }
    oss << "], ";

    // Format swap_track_values list.
    oss << "[";
    for (size_t i = 0; i < log.swap_track.size(); ++i) {
        oss << log.swap_track[i];
        if (i != log.swap_track.size() - 1)
            oss << "; ";
    }
    oss << "],";

    // Format cr_values list.
    oss << "[";
    for (size_t i = 0; i < log.cr.size(); ++i) {
        oss << log.cr[i];
        if (i != log.cr.size() - 1)
            oss << "; ";
    }
    oss << "],";

    // Format delta_S_values list.
    oss << "[";
    for (size_t i = 0; i < log.delta_S_values.size(); ++i) {
        oss << log.delta_S_values[i];
        if (i != log.delta_S_values.size() - 1)
            oss << "; ";
    }
    oss << "]";

    return oss.str();
}
