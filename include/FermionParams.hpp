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
#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"
#include "HMC_Params.hpp"

namespace klft {

// Parameters specific to the Dirac operator
template <size_t rank>
struct diracParams {
  const real_t kappa;
  const IndexArray<rank> dimensions;
  diracParams(const IndexArray<rank> _dimensions, const real_t& _kappa)
      : dimensions(_dimensions),

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
auto getDiracParams(const IndexArray<rank>& dimensions,
                    const FermionMonomial_Params& fparams) {
  diracParams<rank> dParams(dimensions, fparams.kappa);
  return dParams;
}

}  // namespace klft
