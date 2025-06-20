#pragma once
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"

namespace klft {

// Parameters specific to the Dirac operator
template <size_t _rank, size_t _Nc, size_t _RepDim>
struct diracParams {
  static constexpr size_t rank = _rank;
  static constexpr size_t Nc = _Nc;
  static constexpr size_t RepDim = _RepDim;
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const GammaMat<RepDim> gamma5;
  const real_t kappa;
  const IndexArray<rank> dimensions;
  diracParams(const IndexArray<rank> _dimensions,
              const VecGammaMatrix& _gammas,
              const GammaMat<RepDim>& _gamma5,
              const real_t& _kappa)
      : dimensions(_dimensions),
        gammas(_gammas),
        gamma5(_gamma5),
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
  FermionParams(size_t _rank,
                size_t _Nc,
                size_t _RepDim,
                real_t _kappa,
                real_t _tol,
                const std::string& _fermion_type = "Wilson")
      : rank(_rank),
        Nc(_Nc),
        RepDim(_RepDim),
        kappa(_kappa),
        tol(_tol),
        fermion_type(_fermion_type) {}
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

}  // namespace klft
