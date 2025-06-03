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

// this file tests and benchmarks the metropolis kernel for different
// 2D, 3D and 4D gauge fields for U(1), SU(2) and SU(3) gauge groups

#include <iomanip>
#include "Metropolis.hpp"
#include "WilsonFlow.hpp"
#include "GaugePlaquette.hpp"
#include <getopt.h>

#define HLINE "====================================================================\n"

using namespace klft;

int run_test(const size_t stream_size_array) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY") ?
                        std::atoi(std::getenv("KLFT_VERBOSITY")) : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning = std::getenv("KLFT_TUNING") ?
                     std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);
  // 4D volume of the gauge fields
  const real_t volume4D = (real_t)stream_size_array * (real_t)stream_size_array *
                        (real_t)stream_size_array * (real_t)stream_size_array;

  // acceptance rate, output variable from metropolis
  real_t acc_rate = 0.0;

  // timer
  Kokkos::Timer timer;

  // metropolis time
  real_t metropolisTime = std::numeric_limits<real_t>::max();

  printf("Starting benchmark...\n");
  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);
  printf("Reports fastest timing per kernel\n");  
  printf("Lattice extent: %ld\n", stream_size_array);
  printf(HLINE);
  printf("Define Metropolis parameters...\n");
  MetropolisParams params;
  params.Nc = 1;
  params.print();
  printf(HLINE);

  using RNG = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
  RNG rng(params.seed);

  // set Nc = 2
  params.Nc = 2;

  printf("SU(2) Gauge 4D: \n");
  const real_t SU2_field_4D = volume4D * 2.0 * 2.0;
  const real_t SU2_gauge_4D = 4.0 * SU2_field_4D;
  printf("- Size of SU(2) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU2_gauge_4D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");
  
  {
  deviceGaugeField<4, 2> dev_g_SU2_4D(stream_size_array, stream_size_array,
                                        stream_size_array, stream_size_array,
                                        identitySUN<2>());

  printf("Plaq before: %11.4e\n", GaugePlaquette<4, 2>(dev_g_SU2_4D));

  for (int asdf = 0; asdf < 15; ++asdf){
  printf("Sweep Metropolis %d times to get a config\n", STREAM_NTIMES);

  // run the benchmark
  for (index_t k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    acc_rate = sweep_Metropolis<4, 2>(dev_g_SU2_4D, params, rng);
    Kokkos::fence();
    metropolisTime = std::min(metropolisTime, timer.seconds());
  }
  printf(HLINE);
  printf("Latest Acceptance Rate: %11.4e\n", acc_rate);
  printf("Metropolis Kernel Time:     %11.4e s\n", metropolisTime);
  printf("Plaq after: %11.4e\n", GaugePlaquette<4, 2>(dev_g_SU2_4D));
  printf(HLINE);

  WilsonFlowParams wParams;
  wParams.n_steps = 5;
  wParams.eps = 0.0001;
  printf("Plaq after wflow: ");
  for (index_t step = 0; step <20; ++step){
  auto wFlow = WilsonFlow<4, 2, GaugeFieldKind::Standard>(wParams, dev_g_SU2_4D);  
  wFlow.flow();
  Kokkos::fence();
  printf("%11.4e ", GaugePlaquette<4, 2>(dev_g_SU2_4D));
  //printf(HLINE);
  //printf(HLINE); 
  }
  printf("\n");
  printf(HLINE);
  }
  }
  return 0;
}

int test_eps_scan(const size_t lattice_size, klft::real_t flow_time = 0.3, int n_configs = 5) {
  using RNG = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
  RNG rng(12345);

  std::vector<klft::real_t> eps_values = {0.0005, 0.001, 0.002, 0.005, 0.01};
  const index_t Nc = 2;

  printf(HLINE);
  printf("Starting Wilson Flow eps scan\n");
  printf("Target flow time: %g\n", flow_time);
  printf("Lattice size: %zu^4, Nc = %d, Configs per eps = %d\n", lattice_size, Nc, n_configs);
  printf("Scanning %zu eps values...\n", eps_values.size());
  printf(HLINE);

  for (auto eps : eps_values) {
    index_t n_steps = static_cast<index_t>(flow_time / eps);
    std::ostringstream fname;
    fname << "wflow_eps_" << std::fixed << std::setprecision(6) << eps << ".dat";
    std::ofstream fout(fname.str());
printf("[eps = %.6f] -> n_steps = %d --> Output: %s\n", eps, n_steps, fname.str().c_str()); fout << "# Wilson Flow Plaquette test\n"; fout << "# eps = " << eps << ", n_steps = " << n_steps << ", flow_time ~ " << flow_time << "\n";
    fout << "# ConfigID\tFlowTime\tPlaquette\n";

    for (int cfg_id = 0; cfg_id < n_configs; ++cfg_id) {
      printf("  [cfg %d/%d] Metropolis + Flow...", cfg_id + 1, n_configs);
      fflush(stdout);

      MetropolisParams mparams;
      mparams.Nc = Nc;

      deviceGaugeField<4, Nc> gauge(lattice_size, lattice_size, lattice_size, lattice_size,
                                    identitySUN<Nc>());

      for (int sweep = 0; sweep < 15; ++sweep)
        sweep_Metropolis<4, Nc>(gauge, mparams, rng);

      WilsonFlowParams wfparams;
      wfparams.eps = eps;
      wfparams.beta = mparams.beta;
      wfparams.n_steps = 10;

      auto flow = WilsonFlow<4, Nc, GaugeFieldKind::Standard>(wfparams, gauge);
      Kokkos::fence();

      size_t intermediate_step {0};
      klft::real_t plaq = GaugePlaquette<4, Nc>(gauge);
      Kokkos::fence();
      fout << cfg_id << "\t" << real_t(eps * intermediate_step*10*eps) << "\t" << plaq << "\n";
      fout.flush();
      while (intermediate_step*10 < n_steps){ 
      flow.flow();
      Kokkos::fence();
      klft::real_t plaq = GaugePlaquette<4, Nc>(gauge);
      Kokkos::fence();
      ++intermediate_step;
      fout << cfg_id << "\t" << real_t(eps * intermediate_step*10) << "\t" << plaq << "\n";
      fout.flush();
     }

      //printf(" done. Plaquette = %.6f\n", plaq);
    }

    fout.close();
    printf("[eps = %.6f] Finished. Results written to %s\n", eps, fname.str().c_str());
    printf(HLINE);
  }

  printf("All eps scans complete.\n");
  return 0;
}


int parse_args(int argc, char **argv, size_t &stream_array_size) {
  // Defaults
  stream_array_size = 12;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Length of the gauge field dimensions.\n"
      "     Default: 12\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'n': stream_array_size = atoi(optarg); break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0: break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("SU(N) Wilson Flow eps scan test\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  size_t stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = test_eps_scan(stream_array_size);
  } else if (rc == -2) {
    rc = 0;
  }
  Kokkos::finalize();
  return rc;
}
