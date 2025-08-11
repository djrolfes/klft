#include "PTBC.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HamiltonianField.hpp"
#include "InputParser.hpp"
#include "SimulationLogging.hpp"
#include <mpi.h>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int PTBC_execute(const std::string &input_file,
                 const std::string &output_directory) {
  index_t rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank %d of %d\n", rank, size);

  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  bool inputFileParsedCorrectly = (parseInputFile(input_file, gaugeObsParams) &&
                                   parseInputFile(input_file, hmcParams) &&
                                   parseInputFile(input_file, simLogParams) &&
                                   parseInputFile(input_file, ptbcParams));
  if (!inputFileParsedCorrectly) {
    printf("Error parsing input file\n");
    return -1;
  }

  if (ptbcParams.defects.size() != size) {
    printf("Error: Number of defects (%zu) does not match size (%d)\n",
           ptbcParams.defects.size(), size);
    return -1;
  }
  ptbcParams.defect_value = ptbcParams.defects[(rank)];

  ptbcParams.hmc_params = hmcParams;
  RNGType rng(hmcParams.seed + rank);
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);

  if (hmcParams.coldStart) {
    if (hmcParams.Ndims == 4) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(ptbcParams, rng, dist, mt)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<4,
      //   GaugeFieldKind::PTBC>; using DAdjFieldType = DeviceAdjFieldType<4,
      //   3>; run_PTBC<DGaugeFieldType, DAdjFieldType>(
      // ptbcParams, rng, dist, mt);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 3D
    else if (hmcParams.Ndims == 3) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<3, 3,
      //   GaugeFieldKind::PTBC>; using DAdjFieldType = DeviceAdjFieldType<3,
      //   3>; run_PTBC<DGaugeFieldType, DAdjFieldType>(
      // ptbcParams, rng, dist, mt);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 2D
    else if (hmcParams.Ndims == 2) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<2, 3,
      //   GaugeFieldKind::PTBC>; using DAdjFieldType = DeviceAdjFieldType<2,
      //   3>; run_PTBC<DGaugeFieldType, DAdjFieldType>(
      // ptbcParams, rng, dist, mt);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
  } else {
    if (hmcParams.Ndims == 4) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<4, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<4, 3,
      //   GaugeFieldKind::PTBC>; using DAdjFieldType = DeviceAdjFieldType<4,
      //   3>; run_PTBC<DGaugeFieldType, DAdjFieldType>(
      // ptbcParams, rng, dist, mt);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 3D
    else if (hmcParams.Ndims == 3) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<3, 3,
      //   GaugeFieldKind::PTBC>; using DAdjFieldType = DeviceAdjFieldType<3,
      //   3>; run_PTBC<DGaugeFieldType, DAdjFieldType>(
      // ptbcParams, rng, dist, mt);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 2D
    else if (hmcParams.Ndims == 2) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 1, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 2, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        run_PTBC<DGaugeFieldType, DAdjFieldType>(ptbcParams, rng, dist, mt);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<2, 3,
      //   GaugeFieldKind::PTBC>; using DAdjFieldType = DeviceAdjFieldType<2,
      //   3>; run_PTBC<DGaugeFieldType, DAdjFieldType>(
      // ptbcParams, rng, dist, mt);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
  }
  // if tuning is enabled, write the cache file
  // if (KLFT_TUNING) {
  //   const char *cache_file = std::getenv("KLFT_CACHE_FILE");
  //   if (cache_file) {
  //     writeTuneCache(cache_file);
  //   } else {
  //     printf("KLFT_CACHE_FILE not set\n");
  //   }
  // }
  return 0;
}
} // namespace klft
