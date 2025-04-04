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

// a simple test and benchmark for the gauge plaquette

#include "../../include/GaugePlaquette.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <limits>

#include <sys/time.h>

#define Nd 4
#define Nc 1

#define HLINE "=========================================================\n"

using namespace klft;

// init value
constexpr complex_t g_init(1.0, 1.0);

int run_benchmark(const size_t stream_array_size) {
  printf("Reports fastest timing per kernel\n");
  
  const real_t nelem = (real_t)stream_array_size*
                       (real_t)stream_array_size*
                       (real_t)stream_array_size*
                       (real_t)stream_array_size;

  const real_t suN_nelem = nelem*Nc*Nc;

  const real_t gauge_nelem = Nd*suN_nelem;

  printf(HLINE);

  printf("Memory Sizes:\n");
  printf("- Gauge Array Size:                            %d*%d*%" PRIu64 "^4\n",
         Nd, Nc,
         static_cast<uint64_t>(stream_array_size));
  printf("- Size of GaugeField:                          %12.2f MB\n",
         1.0e-6 * gauge_nelem * (real_t)sizeof(complex_t));
  printf("- Total Memory Use (1 GaugeField + 1 Field):   %12.2f MB\n",
         1.0e-6 * (nelem + gauge_nelem) * (real_t)sizeof(complex_t));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);

  printf(HLINE);

  real_t plaquetteTime  = std::numeric_limits<real_t>::max();

  printf("Initializing Gauge...\n");

  deviceGaugeField<Nd,Nc> dev_g(stream_array_size,stream_array_size,stream_array_size,stream_array_size,g_init);

  printf("Starting benchmark...\n");

  Kokkos::Timer timer;

  real_t plaq_value = 0.0;

  for (index_t k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    plaq_value = GaugePlaquette<Nd,Nc>(dev_g);
    Kokkos::fence();
    plaquetteTime = std::min(plaquetteTime, timer.seconds());
  }

  printf(HLINE);

  printf("Plaquette value: %11.4e\n", plaq_value / (nelem * (Nd * (Nd - 1) / 2) * Nc));
  printf("Expected value:  %11.4e\n", 4.0);

  int rc = 0;

  printf(HLINE);

  printf("Plaquette Kernel Time:     %11.4e s\n", plaquetteTime);

  printf("Plaquette BW:              %11.4f GB/s\n",
         1.0e-09 * 1.0 * (real_t)sizeof(complex_t) * (nelem + gauge_nelem) / plaquetteTime);

  printf("Plaquette FLOPS:           %11.4f Gflop/s\n",
         1.0e-09 * nelem *
           6.0 *                 // six plaquettes in 4D      
         ( (2 * 1 * (2 + 4)) +  // two u1 multiplications
           (1 * (2 + 4))     +  // one u1 multiplcation (diagonal elements only)
           (1) )                 // trace accumulation (our plaquette is complex but we're interested only in the real part)
         / plaquetteTime);

  printf(HLINE);

  return rc;
}

int parse_args(int argc, char **argv, size_t &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create 4D GaugeField containing [4][1][1]<N>^4 elements.\n"
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
  printf("U(1) GaugeField 4D plaquette kernel test and benchmark\n");
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