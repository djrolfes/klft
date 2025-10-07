#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeAction.hpp"
#include "GaugeObservable.hpp"
#include "GaugePlaquette.hpp"
#include "Kokkos_Atomic.hpp"

namespace klft {
// template <typename DGaugeFieldType>
// void get_sp_distribution(const typename DGaugeFieldType::type gauge_field,
//                          std::vector<real_t> &rtn, real_t max = 0.1,
//                          real_t bin_width = 0.001) {
//   static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
//   static_assert(Nd == 4);
//   static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
//   static const GaugeFieldKind k =
//       DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
//   using GaugeFieldType = typename DGaugeFieldType::type;
//   using FieldType = typename DeviceFieldType<Nd>::type;
//   FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));
//
//   // number of bins
//   size_t nbins = static_cast<size_t>(max / bin_width);
//   rtn.resize(nbins, 0.0);
//
//   // 1. Create a Kokkos View on the device for the histogram.
//   // Initialize it to zeros.
//   Kokkos::View<real_t *> rtn_d("rtn_d", nbins);
//   Kokkos::deep_copy(rtn_d, 0.0);
//
//   GaugePlaq<Nd, Nc, k> GPlaq(gauge_field, plaq_per_site,
//                              gauge_field.dimensions);
//
//   // tune_and_launch_for<Nd>("compute plaq_per_site", IndexArray<Nd>{0},
//   //                         gauge_field.dimensions, GPlaq);
//   // Kokkos::fence();
//
//   tune_and_launch_for<Nd>(
//       "binning sp's", IndexArray<Nd>{0}, gauge_field.dimensions,
//       KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3) {
//         for (index_t mu = 0; mu < Nd; ++mu) {
//           for (index_t nu = 0; nu < Nd; ++nu) {
//             if (nu > mu) {
//               real_t s = Kokkos::real(2.0 - GPlaq(mu, nu, i0, i1, i2, i3));
//               // now find the end bin:
//               index_t endbin =
//                   static_cast<index_t>(nbins * Kokkos::min(1.0, s / max));
//               endbin = endbin =
//                   0 ? 1 : endbin; // ensure at least one bin is filled
//               for (index_t i = 0; i < endbin; i++) {
//                 Kokkos::atomic_inc(&rtn_d[i]);
//               }
//             }
//           }
//         }
//       });
//   Kokkos::fence();
//
//   Kokkos::View<real_t *>::HostMirror rtn_h =
//   Kokkos::create_mirror_view(rtn_d);
//
//   // Perform the device-to-host copy.
//   Kokkos::deep_copy(rtn_h, rtn_d);
//
//   // 4. Copy data from the host mirror into the final std::vector.
//   for (size_t i = 0; i < nbins; ++i) {
//     rtn[i] = rtn_h(i);
//   }
//
//   auto volume = Nd * Nd;
//   for (size_t vol : gauge_field.dimensions) {
//     volume *= vol;
//   }
//   for (index_t i = 0; i < size(rtn); i++) {
//     rtn[i] /= volume;
//   }
// }

template <typename DGaugeFieldType>
real_t get_spmax(const typename DGaugeFieldType::type gauge_field) {
  static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4);
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));

  // number of bins
  GaugePlaq<Nd, Nc, k> GPlaq(gauge_field, plaq_per_site,
                             gauge_field.dimensions);

  // tune_and_launch_for<Nd>("compute plaq_per_site", IndexArray<Nd>{0},
  //                         gauge_field.dimensions, GPlaq);
  // Kokkos::fence();

  real_t rtn = 0.0;
  auto policy = Policy<Nd>({0, 0, 0, 0}, gauge_field.dimensions);
  Kokkos::parallel_reduce(
      "get h (sp_max)", policy,
      KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3,
                    real_t &local_max) {
        // GPlaq(i0, i1, i2, i3);
        real_t s = 0.0;
        for (index_t mu = 0; mu < Nd; ++mu) {
          for (index_t nu = 0; nu < Nd; ++nu) {
            if (nu > mu) {
              real_t tmp = Kokkos::real(2 - GPlaq(mu, nu, i0, i1, i2, i3));
              s = tmp > s ? tmp : s;
            }
          }
        }
        local_max = Kokkos::max(local_max, s);
      },
      Kokkos::Max<real_t>(rtn));
  Kokkos::fence();

  return rtn;
}

template <typename DGaugeFieldType>
real_t get_spavg(const typename DGaugeFieldType::type gauge_field) {
  static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4);
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));

  // number of bins
  GaugePlaq<Nd, Nc, k> GPlaq(gauge_field, plaq_per_site,
                             gauge_field.dimensions);

  // tune_and_launch_for<Nd>("compute plaq_per_site", IndexArray<Nd>{0},
  //                         gauge_field.dimensions, GPlaq);
  // Kokkos::fence();

  real_t rtn = 0.0;
  auto policy = Policy<Nd>({0, 0, 0, 0}, gauge_field.dimensions);
  Kokkos::parallel_reduce(
      "get h (sp_max)", policy,
      KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3, real_t &local) {
        // GPlaq(i0, i1, i2, i3);
        real_t local_avg = 0.0;
        int tmp = 0;
        for (index_t mu = 0; mu < Nd; ++mu) {
          for (index_t nu = 0; nu < Nd; ++nu) {
            if (nu > mu) {
              local_avg += Kokkos::real(2 - GPlaq(mu, nu, i0, i1, i2, i3));
              tmp++;
            }
          }
        }
        local += local_avg / tmp;
      },
      rtn);
  Kokkos::fence();

  Kokkos::printf("gaugefield size: %lu\n", gauge_field.field.size());

  return rtn / gauge_field.field.size();
}

template <typename DGaugeFieldType, typename HMCType>
index_t do_wflowtest(HMCType &hmc, GaugeObservableParams &gaugeObsParams,
                     std::string output_directory) {

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
  output_file_topologicalcharge << std::fixed << std::setprecision(8);
  output_file_actiondensity << std::fixed << std::setprecision(8);
  output_file_actiondensity_clover << std::fixed << std::setprecision(8);
  output_file_actiondensity_0 << std::fixed << std::setprecision(8);
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
      hmc.hamiltonian_field.gauge_field, 2.0, true));
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
        WilsonAction_full<DGaugeFieldType>(wilson_flow.field, 2.0, true));
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

    if (accept) {

      WilsonFlow<DGaugeFieldType> wilson_flow(
          hmc.hamiltonian_field.gauge_field, gaugeObsParams.wilson_flow_params);
      flow_times.push_back(0.0);
      topological_charges.push_back(get_topological_charge<DGaugeFieldType>(
          hmc.hamiltonian_field.gauge_field));
      action_densities.push_back(
          getActionDensity<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));
      action_densities_clover.push_back(
          getActionDensity<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field));
      action_densities_0.push_back(WilsonAction_full<DGaugeFieldType>(
          hmc.hamiltonian_field.gauge_field, 2.0, true));
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
            getActionDensity<DGaugeFieldType>(wilson_flow.field));
        action_densities_0.push_back(
            WilsonAction_full<DGaugeFieldType>(wilson_flow.field, 2.0, true));
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
  return 0;
}

} // namespace klft
