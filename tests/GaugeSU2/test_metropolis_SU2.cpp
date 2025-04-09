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

// a simple test and benchmark for the metropolis

#include "../../include/Metropolis.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <limits>

#include <sys/time.h>

#define ND 4
#define NC 2

#define HLINE "=========================================================\n"

using namespace klft;

// RNG
#include <Kokkos_Random.hpp>
using RNG = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

int run_benchmark(const size_t stream_array_size) {
  // set verbosity to debug
  setVerbosity(5);
  // enable tuning
  setTuning(true);

  printf("Reports fastest timing per kernel\n");
  
  const real_t nelem = (real_t)stream_array_size*
                       (real_t)stream_array_size*
                       (real_t)stream_array_size*
                       (real_t)stream_array_size;

  const real_t suN_nelem = nelem*NC*NC;

  const real_t gauge_nelem = ND*suN_nelem;

  printf(HLINE);

  printf("Memory Sizes:\n");
  printf("- Gauge Array Size:                            %d*%d*%" PRIu64 "^4\n",
         ND, NC,
         static_cast<uint64_t>(stream_array_size));
  printf("- Size of GaugeField:                          %12.2f MB\n",
         1.0e-6 * gauge_nelem * (real_t)sizeof(complex_t));
  printf("- Total Memory Use (1 GaugeField + 1 staple Field):   %12.2f MB\n",
         1.0e-6 * (2.0 * gauge_nelem) * (real_t)sizeof(complex_t));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);

  printf(HLINE);

  real_t metropolisTime  = std::numeric_limits<real_t>::max();

  printf("Initializing Metropolis...\n");

  RNG rng(12432);

  // create a metropolis params object
  MetropolisParams params;
  params.Ndims = ND;
  params.Nc = NC;
  params.L0 = stream_array_size;
  params.L1 = stream_array_size;
  params.L2 = stream_array_size;
  params.L3 = stream_array_size;
  params.nHits = 10;
  params.beta = 1.0;
  params.delta = 0.1;

  // print the parameters
  params.print();

  printf(HLINE);

  printf("Tuning run...\n");

  // tuning run
  {
  deviceGaugeField<ND,NC> dev_g_tune(stream_array_size,stream_array_size,stream_array_size,stream_array_size,rng,0.1);
  size_t temp = sweep_Metropolis<ND,NC>(dev_g_tune, params, rng, false);
  temp += sweep_Metropolis<ND,NC>(dev_g_tune, params, rng, true);
  Kokkos::fence();
  }

  printf(HLINE);

  printf("Initializing Gauge...\n");

  deviceGaugeField<ND,NC> dev_g(stream_array_size,stream_array_size,stream_array_size,stream_array_size,rng,0.1);

  printf(HLINE);

  printf("Starting benchmark...\n");

  Kokkos::Timer timer;

  size_t nAcc = 0;

  for (index_t k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    nAcc = sweep_Metropolis<ND,NC>(dev_g, params, rng, false);
    nAcc += sweep_Metropolis<ND,NC>(dev_g, params, rng, true);
    Kokkos::fence();
    metropolisTime = std::min(metropolisTime, timer.seconds());
  }

  int rc = 0;

  printf(HLINE);

  printf("Metropolis Kernel Time:     %11.4e s\n", metropolisTime);

  printf(HLINE);

  return rc;
}

int parse_args(int argc, char **argv, size_t &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create 4D GaugeField containing [4][2][2]<N>^4 elements.\n"
      "     Default: 32\n"
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
  printf("SU(2) GaugeField 4D metropolis kernel test and benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  size_t stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_benchmark(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}