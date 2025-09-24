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
    if (mu == nu || mu == rho || mu == sigma || nu == rho || nu == sigma ||
        rho == sigma) {
      return 0;
    }
    // pack into array
    constexpr int N = 4;
    IndexArray<N> idx{mu, nu, rho, sigma};
    // count inversions
    int inv = 0;
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        if (idx[i] > idx[j])
          ++inv;
      }
    }
    // parity of inversion count gives sign
    return (inv % 2 == 0 ? +1 : -1);
  }

  // TODO: make epsilon resolve at compiletime for the relevant combinations

  // now define the topological charge calculation for a single site (should
  // this also be parallized over some directions?)
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) const {
    real_t local_charge{0.0};
    RealMatrix C1, C2;
    Kokkos::Array<Kokkos::Array<RealMatrix, Nd>, Nd> C;

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        // get the clover C_munu
        C[mu][nu] = (fst(FSTTag{}, i0, i1, i2, i3, mu, nu));
        C[nu][mu] = (fst(FSTTag{}, i0, i1, i2, i3, nu, mu));
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
            local_charge += epsilon4(mu, nu, rho, sigma) *
                            trace(C[mu][nu] * C[rho][sigma]); //
            local_charge +=
                epsilon4(nu, mu, rho, sigma) * trace(C[nu][mu] * C[rho][sigma]);
            local_charge +=
                epsilon4(mu, nu, sigma, rho) * trace(C[mu][nu] * C[sigma][rho]);
            local_charge +=
                epsilon4(nu, mu, sigma, rho) * trace(C[nu][mu] * C[sigma][rho]);
          }
        }
      }
    }
    // charge_per_site(i0, i1, i2, i3) = local_charge / 16;
  }

  // return the clover C_munu
  // template <typename indexType>
  // KOKKOS_FORCEINLINE_FUNCTION RealMatrix
  // get_clover(const indexType i0, const indexType i1, const indexType i2,
  //            const indexType i3, index_t mu, index_t nu) const {
  //
  //   SUN<Nc> P_munu = zeroSUN<Nc>();
  //   const IndexArray<Nd> x{static_cast<index_t>(i0),
  //   static_cast<index_t>(i1),
  //                          static_cast<index_t>(i2),
  //                          static_cast<index_t>(i3)};
  //
  //   // Helper lambda for modular arithmetic
  //   auto mod = [&](index_t s, index_t dir) {
  //     return (s + this->dimensions[dir]);
  //   };
  //
  //   // 1. Plaquette in (+mu, +nu) plane starting at x
  //   IndexArray<Nd> x_p_mu = x;
  //   x_p_mu[mu] = (x[mu] + 1) % dimensions[mu];
  //   IndexArray<Nd> x_p_nu = x;
  //   x_p_nu[nu] = (x[nu] + 1) % dimensions[nu];
  //   P_munu += g_in(x, mu) * g_in(x_p_mu, nu) * conj(g_in(x_p_nu, mu)) *
  //             conj(g_in(x, nu));
  //
  //   // 2. Plaquette in (+mu, -nu) plane starting at x
  //   IndexArray<Nd> x_m_nu = x;
  //   x_m_nu[nu] = mod(x[nu] - 1, nu) % dimensions[nu];
  //   IndexArray<Nd> x_p_mu_m_nu = x_p_mu;
  //   x_p_mu_m_nu[nu] = mod(x_p_mu[nu] - 1, nu) % dimensions[nu];
  //   P_munu += g_in(x, mu) * conj(g_in(x_p_mu_m_nu, nu)) *
  //             conj(g_in(x_m_nu, mu)) * g_in(x_m_nu, nu);
  //
  //   // 3. Plaquette in (-mu, +nu) plane starting at x
  //   IndexArray<Nd> x_m_mu = x;
  //   x_m_mu[mu] = mod(x[mu] - 1, mu) % dimensions[mu];
  //   IndexArray<Nd> x_m_mu_p_nu = x_m_mu;
  //   x_m_mu_p_nu[nu] = (x_m_mu[nu] + 1) % dimensions[nu];
  //   P_munu += conj(g_in(x_m_mu, mu)) * g_in(x_m_mu, nu) *
  //             g_in(x_m_mu_p_nu, mu) * conj(g_in(x, nu));
  //
  //   // 4. Plaquette in (-mu, -nu) plane starting at x
  //   IndexArray<Nd> x_m_mu_m_nu = x_m_mu;
  //   x_m_mu_m_nu[nu] = mod(x_m_mu[nu] - 1, nu) % dimensions[nu];
  //   P_munu += conj(g_in(x_m_mu, mu)) * conj(g_in(x_m_mu_m_nu, nu)) *
  //             g_in(x_m_mu_m_nu, mu) * g_in(x_m_nu, nu);
  //
  //   return imag(P_munu) * 0.25;
  // }
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

  real_t charge = TCharge.charge_per_site.sum();
  Kokkos::fence();
  // charge /= 32 * PI * PI;

  return charge / (32 * PI * PI);
}
} // namespace klft
