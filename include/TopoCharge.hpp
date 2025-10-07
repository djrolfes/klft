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
#include "FieldStrengthTensor.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_Macros.hpp"
#include <unistd.h>

namespace klft {
// first define the necessary functor

template <typename DGaugeFieldType,
          typename FSTTag =
              typename FieldStrengthTensor<DGaugeFieldType>::CloverDef>
struct TopoCharge {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4,
                "Topological charge is only defined for 4D gauge fields.");
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  // define the gauge field type
  using GaugeFieldType = typename DGaugeFieldType::type;
  const GaugeFieldType g_in;
  // define the field type
  using FieldType = typename DeviceScalarFieldType<Nd>::type;
  FieldType charge_per_site;

  const FieldStrengthTensor<DGaugeFieldType> fst;

  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;
  using ComplexMatrix = Kokkos::Array<Kokkos::Array<complex_t, Nc>, Nc>;

  // define the dimensions of the given Field
  const IndexArray<Nd> dimensions;

  TopoCharge(const GaugeFieldType g_in)
      : g_in(g_in), dimensions(g_in.dimensions),
        charge_per_site(g_in.dimensions, real_t(0)), fst(g_in) {}

  /// Return the 4-D Levi–Civita symbol ε_{μνρσ} for indices 0..3.
  /// Zero if any indices repeat; +1 or –1 otherwise.
  // This should be evaluated at compile time.
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr int epsilon4(int mu, int nu, int rho, int sigma) const {
    // repeats -> 0
    // if (mu == nu || mu == rho || mu == sigma || nu == rho || nu == sigma ||
    //     rho == sigma) {
    //   return 0;
    // }
    index_t parity = (mu - nu) * (mu - rho) * (mu - sigma) * (nu - rho) *
                     (nu - sigma) * (rho - sigma);

    // parity of inversion count gives sign
    return parity > 0 ? (parity == 0 ? 0 : 1) : -1;
  }

  // now define the topological charge calculation for a single site (should
  // this also be parallized over some directions?)
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) const {
    real_t local_charge{0.0};
    ComplexMatrix C1, C2;
    Kokkos::Array<Kokkos::Array<SUNAdj<Nc>, Nd>, Nd> C;

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        // get the clover C_munu
        C[mu][nu] = (fst(FSTTag{}, i0, i1, i2, i3, mu, nu));
        // C[nu][mu] = (fst(FSTTag{}, i0, i1, i2, i3, nu, mu));
      }
    }

    // TODO 12.05.: implement this according to 1708.00696
    // local_charge += (trace(fst(FSTTag{}, i0, i1, i2, i3, 0, 1) *
    //                        fst(FSTTag{}, i0, i1, i2, i3, 2, 3)) +
    //                  trace(fst(FSTTag{}, i0, i1, i2, i3, 0, 2) *
    //                        fst(FSTTag{}, i0, i1, i2, i3, 3, 1)) +
    //                  trace(fst(FSTTag{}, i0, i1, i2, i3, 0, 3) *
    //                        fst(FSTTag{}, i0, i1, i2, i3, 1, 2)));
    charge_per_site(i0, i1, i2, i3) = local_charge;
#pragma unroll
    for (int mu = 0; mu < Nd - 1; ++mu) {
#pragma unroll
      for (int nu = mu + 1; nu < Nd; ++nu) {
        if (mu == nu)
          continue;
#pragma unroll
        for (int rho = 0; rho < Nd - 1; ++rho) {
          if (rho == mu || rho == nu)
            continue;
#pragma unroll
          for (int sigma = rho + 1; sigma < Nd; ++sigma) {
            if (sigma == mu || sigma == nu)
              continue;
#pragma unroll
            for (int i = 0; i < NcAdj<Nc>; ++i) {
              local_charge += epsilon4(mu, nu, rho, sigma) *
                              (C[mu][nu][i] * C[rho][sigma][i]);
            }
          }
        }
      }
    }
    charge_per_site(i0, i1, i2, i3) = local_charge;
    // charge_per_site(i0, i1, i2, i3) = local_charge / 16;
  }
};

template <typename DGaugeFieldType>
real_t get_topological_charge(const typename DGaugeFieldType::type g_in) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>(),
                "get_topological_charge requires a device gauge field type.");
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4,
                "Topological charge is only defined for 4D gauge fields.");

  DEBUG_MPI_PRINT("enter get_topological_charge");
  // define the functor
  TopoCharge<DGaugeFieldType> TCharge(g_in);
  tune_and_launch_for<Nd>("Calculate topological charge", IndexArray<Nd>{0},
                          g_in.dimensions, TCharge);
  Kokkos::fence();

  real_t charge = -4.0 * TCharge.charge_per_site.sum();
  Kokkos::fence();
  // charge /= 32 * PI * PI;

  return charge / (-32 * PI * PI);
}
} // namespace klft
