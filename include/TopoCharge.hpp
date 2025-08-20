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

// define plaquette functions for different gauge fields

#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_Macros.hpp"
#include <unistd.h>

namespace klft {
// first define the necessary functor
//

template <typename T, size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<T, N>, N>
operator*(const Kokkos::Array<Kokkos::Array<T, N>, N> &a,
          const Kokkos::Array<Kokkos::Array<T, N>, N> &b) {
  Kokkos::Array<Kokkos::Array<T, N>, N> c;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      c[i][j] = a[i][0] * b[0][j];
#pragma unroll
      for (size_t k = 1; k < N; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

template <typename T, size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<T, N>, N>
operator-(const Kokkos::Array<Kokkos::Array<T, N>, N> &a) {
  Kokkos::Array<Kokkos::Array<T, N>, N> c;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      c[i][j] = -a[i][j];
    }
  }
  return c;
}
//

template <typename DGaugeFieldType> struct TopoCharge {
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

  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;

  // define the dimensions of the given Field
  const IndexArray<Nd> dimensions;

  TopoCharge(const GaugeFieldType &g_in)
      : g_in(g_in), dimensions(g_in.dimensions),
        charge_per_site(g_in.dimensions, real_t(0)) {}

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
    int rho;
    int sigma;
    int mu = 0;
    RealMatrix C1, C2;
    Kokkos::Array<Kokkos::Array<RealMatrix, Nd>, Nd> C;

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        // get the clover C_munu
        RealMatrix C_munu = get_clover(i0, i1, i2, i3, mu, nu);
        C[mu][nu] = C_munu;
        C[nu][mu] = -C_munu;
      }
    }

// TODO 12.05.: implement this according to 1708.00696
#pragma unroll
    for (int mu = 0; mu < 4; ++mu) {
#pragma unroll
      for (int nu = 0; nu < 4; ++nu) {
#pragma unroll
        for (int rho = 0; rho < 4; ++rho) {
#pragma unroll
          for (int sigma = 0; sigma < 4; ++sigma) {
            if (epsilon4(mu, nu, rho, sigma) == 0) {
              continue;
            }
            local_charge +=
                epsilon4(mu, nu, rho, sigma) * trace(C[mu][nu] * C[rho][sigma]);
          }
        }
      }
    }
    charge_per_site(i0, i1, i2, i3) = local_charge / 24;
    // charge_per_site(i0, i1, i2, i3) = local_charge / 16;
  }

  // get the imaginary parts of an SUN matrix
  KOKKOS_FORCEINLINE_FUNCTION RealMatrix imag(const SUN<Nc> &in) const {
    RealMatrix out{0};
#pragma unroll
    for (int i = 0; i < Nc; ++i) {
#pragma unroll
      for (int j = 0; j < Nc; ++j) {
        out[i][j] = in[i][j].imag();
      }
    }
    return out;
  }

  // return the trace of a RealMatrix
  KOKKOS_FORCEINLINE_FUNCTION real_t trace(const RealMatrix &in) const {
    real_t out{0};
#pragma unroll
    for (int i = 0; i < Nc; ++i) {
      out += in[i][i];
    }
    return out;
  }

  // return the clover C_munu
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION RealMatrix
  get_clover(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3, index_t mu, index_t nu) const {

    IndexArray<4> site1{static_cast<index_t>(i0), static_cast<index_t>(i1),
                        static_cast<index_t>(i2), static_cast<index_t>(i3)};
    IndexArray<4> site2{static_cast<index_t>(i0), static_cast<index_t>(i1),
                        static_cast<index_t>(i2), static_cast<index_t>(i3)};
    IndexArray<4> site3{static_cast<index_t>(i0), static_cast<index_t>(i1),
                        static_cast<index_t>(i2), static_cast<index_t>(i3)};

    site2[mu] = (site2[mu] + 1) % this->dimensions[mu];
    site3[nu] = (site3[nu] + 1) % this->dimensions[nu];
    SUN<Nc> P = this->g_in(site1, mu) * conj(this->g_in(site2, nu)) *
                conj(this->g_in(site3, mu)) * this->g_in(site1, nu);

    site2[mu] =
        (site2[mu] - 2 + this->dimensions[mu]) % this->dimensions[mu]; // -mu
    site3[mu] = (site3[mu] - 1 + this->dimensions[mu]) %
                this->dimensions[mu]; // +nu,-mu
    P += this->g_in(site1, nu) * conj(this->g_in(site3, mu)) *
         conj(this->g_in(site2, nu)) * this->g_in(site2, mu);

    site2[mu] = (site2[mu] + 1) % this->dimensions[mu]; // -0
    site2[nu] =
        (site2[nu] - 1 + this->dimensions[nu]) % this->dimensions[nu]; // -nu
    site3[mu] = (site3[mu] + 2) % this->dimensions[mu]; // +nu,+mu
    site3[nu] = (site3[nu] - 2 + this->dimensions[nu]) %
                this->dimensions[nu]; // -nu,+mu
    P += conj(this->g_in(site2, nu)) * conj(this->g_in(site3, nu)) *
         this->g_in(site3, mu) * conj(this->g_in(site1, mu));

    site3[mu] = (site3[mu] - 2 + this->dimensions[mu]) %
                this->dimensions[mu]; // -nu,-mu
    site1[mu] =
        (site1[mu] - 1 + this->dimensions[mu]) % this->dimensions[mu]; // -mu
    P += conj(this->g_in(site1, mu)) * conj(this->g_in(site3, nu)) *
         this->g_in(site3, mu) * this->g_in(site2, nu);
    return imag(P);
  }
};

template <typename DGaugeFieldType>
real_t get_topological_charge(const typename DGaugeFieldType::type &g_in) {
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
  charge /= 32 * PI * PI;

  return charge;
}
} // namespace klft
