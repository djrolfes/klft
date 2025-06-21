#include "../include/InputParser.hpp"
#include "../include/klft.hpp"
#include "AdjointSUN.hpp"
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GaugeObservable.hpp"
#include "HMC_Params.hpp"

namespace klft {
int HMC_execute(const std::string& input_file) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);
  // if tuning is enbled, check if the user has set the
  // KLFT_CACHE_FILE environment variable
  if (tuning) {
    const char* cache_file = std::getenv("KLFT_CACHE_FILE");
    // if it exists, read the cache
    if (cache_file) {
      if (KLFT_VERBOSITY > 0) {
        printf("Reading cache file: %s\n", cache_file);
      }
      readTuneCache(cache_file);
    }
  }
  HMCParams hmcParams;
  // parse the input file for HMC parameters
  if (!parseInputFile(input_file, hmcParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  Integrator_Params integratorParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, integratorParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  FermionMonomial_Params fermionParams;
  if (!parseInputFile(input_file, fermionParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  GaugeMonomial_Params gaugeMonomialParams;
  if (!parseInputFile(input_file, gaugeMonomialParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseSanityChecks(integratorParams, gaugeMonomialParams, fermionParams))
    //   FermionParams fparams;
    //   if (!parseInputFile(input_file, fparams)) {
    //     printf("Error parsing input file\n");
    //     return -1;
    //   }
    // print the parameters
    hmcParams.print();
  integratorParams.print();
  //   gaugeObsParams.print();
  fermionParams.print();
  gaugeMonomialParams.print();

  // Start building the Fields

  return 1;
}
}  // namespace klft
