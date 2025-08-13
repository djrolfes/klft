#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"
#include "HMC_Params.hpp"

namespace klft {

// Parameters specific to the Dirac operator
template <size_t rank, size_t RepDim> struct diracParams {
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const GammaMat<RepDim> gamma5;
  const real_t kappa;
  const IndexArray<rank> dimensions;
  diracParams(const IndexArray<rank> _dimensions, const VecGammaMatrix &_gammas,
              const GammaMat<RepDim> &_gamma5, const real_t &_kappa)
      : dimensions(_dimensions), gammas(_gammas), gamma5(_gamma5),
        kappa(_kappa) {}
};

struct FermionParams {
  std::string fermion_type;
  std::string Solver;
  size_t rank;
  size_t Nc;
  size_t RepDim;
  real_t kappa;
  real_t tol;
  FermionParams(size_t _rank, size_t _Nc, size_t _RepDim, real_t _kappa,
                real_t _tol, const std::string &_fermion_type = "Wilson")
      : rank(_rank), Nc(_Nc), RepDim(_RepDim), kappa(_kappa), tol(_tol),
        fermion_type(_fermion_type) {}
  FermionParams() = default;
  void print() const {
    printf("Fermion Parameter:\n");
    printf("  Fermion Type: %s\n", fermion_type.c_str());
    printf("  Solver: %s\n", Solver.c_str());
    printf("  Rank: %zu\n", rank);
    printf("  Nc: %zu\n", Nc);
    printf("  RepDim: %zu\n", RepDim);
    printf("  Kappa: %f\n", kappa);
    printf("  Tolerance: %f\n", tol);
  }
};
template <size_t rank>
auto getDiracParams(const IndexArray<rank> &dimensions,
                    const FermionMonomial_Params &fparams) {
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

} // namespace klft
