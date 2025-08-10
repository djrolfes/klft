#pragma once
#include <fstream>
#include <iomanip>

#include "FermionParams.hpp"
#include "PionCorrelator.hpp"

namespace klft {
struct FermionObservableParams {
  size_t measurement_interval;
  bool measure_pion_correlator;
  std::vector<std::vector<real_t>> pion_correlator;
  std::string pion_correlator_filename;
  std::vector<size_t> measurement_steps;
  real_t tol;
  real_t kappa;
  size_t RepDim;
  bool write_to_file;

  //
  size_t flush;  // interval to flush measurements to file, 0 to flush at the
  // end of the simulation

  FermionObservableParams()
      : measurement_interval(0),
        measure_pion_correlator(false),
        flush(25),
        tol(10e-8),
        kappa(0.15),
        RepDim(4) {}
};
template <size_t rank>
auto getDiracParams(const IndexArray<rank>& dimensions,
                    const FermionObservableParams& fparams) {
  if (fparams.RepDim == 4) {
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    diracParams<rank, 4> dParams(dimensions, gammas, gamma5, fparams.kappa);
    return dParams;

  } else {
    printf("Warning: Unsupported Gamma Matrix Representation\n");
    printf("Warning: Fallback RepDim = 4\n");
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    diracParams<rank, 4> dParams(dimensions, gammas, gamma5, fparams.kappa);
    return dParams;
  }
}

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT,
                    typename,
                    typename> class _Solver,
          template <typename, typename> class DiracOpT>
void measureFermionObservables(const typename DGaugeFieldType::type& g_in,
                               FermionObservableParams& params,
                               const size_t step) {
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }
  if (KLFT_VERBOSITY > 1) {
    printf("Measurement of Fermion Observables\n");
    printf("step: %zu\n", step);
  }
  if (params.measure_pion_correlator) {
    auto PC =
        PionCorrelator<DSpinorFieldType, DGaugeFieldType, _Solver, DiracOpT>(
            g_in,
            getDiracParams<DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank>(
                g_in.dimensions, params),
            params.tol);
    params.pion_correlator.push_back(PC);
    if (KLFT_VERBOSITY > 1) {
      printf("Pion Correlator:\n");
      for (auto&& i : PC) {
        printf("%f, ", i);
      }
      printf("\n");
    }
  }
  params.measurement_steps.push_back(step);
  return;
}
inline void flushPionCorrelator(std::ofstream& file,
                                const FermionObservableParams& params,
                                const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if plaquette measurements are available
  if (!params.measure_pion_correlator) {
    printf("Error: no plaquette measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, pion correlator\n";
  for (size_t i = 0; i < params.pion_correlator.size(); ++i) {
    file << params.measurement_steps[i] << ", ";
    for (auto&& j : params.pion_correlator[i]) {
      file << j << ",";
    }
    file << "\n";
  }
}

inline void flushAllFermionObservables(const FermionObservableParams& params,
                                       const std::string& output_directory,
                                       const bool HEADER = true,
                                       const int& p = std::cout.precision()) {
  std::setprecision(p);

  if (!params.write_to_file) {
    printf("write_to_file is not enabled\n");
    return;
  }

  if (params.measure_pion_correlator && params.pion_correlator_filename != "") {
    std::ofstream file(output_directory + params.pion_correlator_filename,
                       std::ios::app);
    printf("Flushing Pion correlator");
    flushPionCorrelator(file, params, HEADER);
    file.close();
  }
}

inline void clearAllFermionObservables(FermionObservableParams& params) {
  params.measurement_steps.clear();
  params.pion_correlator.clear();
}

}  // namespace klft
