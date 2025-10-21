#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeAction.hpp"
#include "GaugeObservable.hpp"
#include "GaugePlaquette.hpp"
#include "Kokkos_Atomic.hpp"
#include "SimulationLogging.hpp"

namespace klft {

template <typename DGaugeFieldType, typename HMCType>
index_t do_wflowtest(HMCType &hmc, GaugeObservableParams &gaugeObsParams,
                     SimulationLoggingParams &simLogParams,
                     std::string output_directory) {

  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  // Construct the output filename. Each MPI rank will get its own file.
  std::string output_filename =
      output_directory + "topological_charge_cumulative.txt";
  std::ofstream output_file_topologicalcharge(output_filename);

  if (!output_file_topologicalcharge.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename.c_str());
    return -1; // Or handle the error as appropriate
  }

  std::string output_filename_action_density =
      output_directory + "action_densities_cumulative.txt";
  std::ofstream output_file_actiondensity(output_filename_action_density);

  if (!output_file_actiondensity.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename_action_density.c_str());
    return -1; // Or handle the error as appropriate
  }

  std::string output_filename_action_density_clover =
      output_directory + "action_densities_clover_cumulative.txt";
  std::ofstream output_file_actiondensity_clover(
      output_filename_action_density_clover);

  if (!output_file_actiondensity_clover.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename_action_density_clover.c_str());
    return -1; // Or handle the error as appropriate
  }

  std::string output_filename_action_density_0 =
      output_directory + "action_densities_0_cumulative.txt";
  std::ofstream output_file_actiondensity_0(output_filename_action_density_0);

  if (!output_file_actiondensity_0.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename_action_density_0.c_str());
    return -1; // Or handle the error as appropriate
  }

  std::string output_filename_sp_dist = output_directory + "sp_avg.txt";
  std::ofstream output_file_sp_avg(output_filename_sp_dist);

  if (!output_file_sp_avg.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename_sp_dist.c_str());
    return -1; // Or handle the error as appropriate
  }

  std::string output_filename_sp_max = output_directory + "sp_max.txt";
  std::ofstream output_file_sp_max(output_filename_sp_max);

  if (!output_file_sp_max.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename_sp_max.c_str());
    return -1; // Or handle the error as appropriate
  }

  // Set precision for floating point numbers in the output file
  output_file_topologicalcharge << std::fixed << std::setprecision(12);
  output_file_actiondensity << std::fixed << std::setprecision(12);
  output_file_actiondensity_clover << std::fixed << std::setprecision(12);
  output_file_actiondensity_0 << std::fixed << std::setprecision(12);
  output_file_sp_avg << std::fixed << std::setprecision(12);
  output_file_sp_max << std::fixed << std::setprecision(12);
  bool header_written = false;

  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  index_t flow_steps = gaugeObsParams.wilson_flow_params.n_steps;
  gaugeObsParams.wilson_flow_params.n_steps = 1;
  std::vector<real_t> flow_times;
  std::vector<real_t> topological_charges;
  std::vector<real_t> action_densities;
  std::vector<real_t> action_densities_clover;
  std::vector<real_t>
      action_densities_0; // WilsonAciton rescalled a la PhysRevD.95.094508
  std::vector<real_t> sp_avg;
  std::vector<real_t> sp_max;
  real_t sp_dist_max = 0.1;
  real_t sp_dist_bin_width = 0.001;

  WilsonFlow<DGaugeFieldType> wilson_flow(hmc.hamiltonian_field.gauge_field,
                                          gaugeObsParams.wilson_flow_params);
  flow_times.push_back(0.0);
  topological_charges.push_back(get_topological_charge<DGaugeFieldType>(
      hmc.hamiltonian_field.gauge_field));
  action_densities.push_back(
      getActionDensity<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));
  action_densities_clover.push_back(getActionDensity_clover<DGaugeFieldType>(
      hmc.hamiltonian_field.gauge_field));
  action_densities_0.push_back(WilsonAction_full<DGaugeFieldType>(
      hmc.hamiltonian_field.gauge_field, 2.0 * Nc, true));
  // get_sp_distribution<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field,
  //                                      sp_avg, sp_dist_max,
  //                                      sp_dist_bin_width);
  sp_avg.push_back(
      get_spavg<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));
  sp_max.push_back(
      get_spmax<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));

  for (int flow_Step = 1; flow_Step <= flow_steps; ++flow_Step) {
    // perform wilson flow step
    wilson_flow.flow();

    flow_times.push_back(flow_Step * gaugeObsParams.wilson_flow_params.eps);
    topological_charges.push_back(
        get_topological_charge<DGaugeFieldType>(wilson_flow.field));
    action_densities.push_back(
        getActionDensity<DGaugeFieldType>(wilson_flow.field));
    action_densities_clover.push_back(
        getActionDensity_clover<DGaugeFieldType>(wilson_flow.field));
    action_densities_0.push_back(
        WilsonAction_full<DGaugeFieldType>(wilson_flow.field, 2.0 * Nc, true));
    sp_avg.push_back(get_spavg<DGaugeFieldType>(wilson_flow.field));
    sp_max.push_back(get_spmax<DGaugeFieldType>(wilson_flow.field));

    // measure observables
  }
  // Write the header only once, before the first line of data
  if (!header_written) {
    output_file_topologicalcharge << "hmc_step";
    output_file_actiondensity << "hmc_step";
    output_file_actiondensity_clover << "hmc_step";
    output_file_actiondensity_0 << "hmc_step";
    output_file_sp_avg << "hmc_step";
    output_file_sp_max << "hmc_step";
    for (const auto &t : flow_times) {
      output_file_topologicalcharge << "," << t;
      output_file_actiondensity << "," << t;
      output_file_actiondensity_clover << "," << t;
      output_file_actiondensity_0 << "," << t;
      output_file_sp_max << "," << t;
      output_file_sp_avg << "," << t;
    }
    output_file_sp_avg << "\n";
    output_file_topologicalcharge << "\n";
    output_file_actiondensity << "\n";
    output_file_actiondensity_clover << "\n";
    output_file_actiondensity_0 << "\n";
    output_file_sp_max << "\n";
    header_written = true;
  }

  // Write the data for the current step
  output_file_topologicalcharge << 0;
  for (const auto &charge : topological_charges) {
    output_file_topologicalcharge << "," << charge;
  }
  output_file_topologicalcharge << "\n";
  output_file_actiondensity << 0;
  for (const auto &density : action_densities) {
    output_file_actiondensity << "," << density;
  }
  output_file_actiondensity << "\n";

  output_file_actiondensity_clover << 0;
  for (const auto &density : action_densities_clover) {
    output_file_actiondensity_clover << "," << density;
  }
  output_file_actiondensity_clover << "\n";

  output_file_actiondensity_0 << 0;
  for (const auto &density : action_densities_0) {
    output_file_actiondensity_0 << "," << density;
  }
  output_file_actiondensity_0 << "\n";

  output_file_sp_avg << 0;
  for (const auto &dist : sp_avg) {
    output_file_sp_avg << "," << dist;
  }
  output_file_sp_avg << "\n";

  output_file_sp_max << 0;
  for (const auto &max : sp_max) {
    output_file_sp_max << "," << max;
  }
  output_file_sp_max << "\n";

  // hmc loop
  for (size_t step = 0; step < hmc.params.nsteps; ++step) {
    timer.reset();
    flow_times.clear();
    topological_charges.clear();
    action_densities.clear();
    action_densities_clover.clear();
    action_densities_0.clear();
    sp_avg.clear();
    sp_max.clear();

    // perform hmc_step
    accept = hmc.hmc_step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n", step,
             static_cast<size_t>(accept), acc_rate, time);
    }

    // Simulation logging: log step, delta_H, acceptance rate, accept, time
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    flushSimulationLogs(simLogParams, step, true);

    if (accept) {
      WilsonFlow<DGaugeFieldType> wilson_flow(
          hmc.hamiltonian_field.gauge_field, gaugeObsParams.wilson_flow_params);
      flow_times.push_back(0.0);
      topological_charges.push_back(get_topological_charge<DGaugeFieldType>(
          hmc.hamiltonian_field.gauge_field));
      action_densities.push_back(
          getActionDensity<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));
      action_densities_clover.push_back(
          getActionDensity_clover<DGaugeFieldType>(
              hmc.hamiltonian_field.gauge_field));
      action_densities_0.push_back(WilsonAction_full<DGaugeFieldType>(
          hmc.hamiltonian_field.gauge_field, 2.0 * Nc, true));
      sp_avg.push_back(
          get_spavg<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));
      sp_max.push_back(
          get_spmax<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));

      for (int flow_Step = 1; flow_Step <= flow_steps; ++flow_Step) {
        // perform wilson flow step
        wilson_flow.flow();

        flow_times.push_back(flow_Step * gaugeObsParams.wilson_flow_params.eps);
        topological_charges.push_back(
            get_topological_charge<DGaugeFieldType>(wilson_flow.field));
        action_densities.push_back(
            getActionDensity<DGaugeFieldType>(wilson_flow.field));
        action_densities_clover.push_back(
            getActionDensity_clover<DGaugeFieldType>(wilson_flow.field));
        action_densities_0.push_back(WilsonAction_full<DGaugeFieldType>(
            wilson_flow.field, 2.0 * Nc, true));
        sp_avg.push_back(get_spavg<DGaugeFieldType>(wilson_flow.field));
        sp_max.push_back(get_spmax<DGaugeFieldType>(wilson_flow.field));

        // measure observables
      }

      // Write the data for the current step
      output_file_topologicalcharge << step;
      for (const auto &charge : topological_charges) {
        output_file_topologicalcharge << "," << charge;
      }
      output_file_topologicalcharge << "\n";
      output_file_actiondensity << step;
      for (const auto &density : action_densities) {
        output_file_actiondensity << "," << density;
      }
      output_file_actiondensity << "\n";

      output_file_actiondensity_clover << step;
      for (const auto &density : action_densities_clover) {
        output_file_actiondensity_clover << "," << density;
      }
      output_file_actiondensity_clover << "\n";

      output_file_actiondensity_0 << step;
      for (const auto &density : action_densities_0) {
        output_file_actiondensity_0 << "," << density;
      }
      output_file_actiondensity_0 << "\n";

      output_file_sp_avg << step;
      for (const auto &dist : sp_avg) {
        output_file_sp_avg << "," << dist;
      }
      output_file_sp_avg << "\n";
      output_file_sp_max << step;
      for (const auto &max : sp_max) {
        output_file_sp_max << "," << max;
      }
      output_file_sp_max << "\n";
    }
  }
  // Final flush of simulation logs
  forceflushSimulationLogs(simLogParams, true);
  return 0;
}

} // namespace klft
