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
#include "Kokkos_Macros.hpp"
#include <unistd.h>

namespace klft {

// Operator overloads for RealMatrix
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

template <typename DGaugeFieldType> struct TopoCharge {
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4,
                "Topological charge is only defined for 4D gauge fields.");
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;

  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceScalarFieldType<Nd>::type;
  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;

  const GaugeFieldType g_in;
  FieldType charge_per_site;
  const IndexArray<Nd> dimensions;

  TopoCharge(const GaugeFieldType g_in)
      : g_in(g_in), dimensions(g_in.dimensions),
        charge_per_site(g_in.dimensions, real_t(0)) {}

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr int epsilon4(int mu, int nu, int rho, int sigma) const {
    if (mu == nu || mu == rho || mu == sigma || nu == rho || nu == sigma ||
        rho == sigma) {
      return 0;
    }
    IndexArray<4> idx{mu, nu, rho, sigma};
    int inv = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        if (idx[i] > idx[j])
          ++inv;
      }
    }
    return (inv % 2 == 0 ? +1 : -1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) const {
    real_t local_charge{0.0};
    Kokkos::Array<Kokkos::Array<RealMatrix, Nd>, Nd> C;

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        RealMatrix C_munu = get_clover(i0, i1, i2, i3, mu, nu);
        C[mu][nu] = C_munu;
        C[nu][mu] = -C_munu;
      }
    }

#pragma unroll
    for (int mu = 0; mu < 4; ++mu) {
#pragma unroll
      for (int nu = 0; nu < 4; ++nu) {
#pragma unroll
        for (int rho = 0; rho < 4; ++rho) {
#pragma unroll
          for (int sigma = 0; sigma < 4; ++sigma) {
            int epsilon = epsilon4(mu, nu, rho, sigma);
            if (epsilon == 0)
              continue;
            local_charge += epsilon * trace(C[mu][nu] * C[rho][sigma]);
          }
        }
      }
    }
    charge_per_site(i0, i1, i2, i3) = local_charge / 16.0;
  }

  // --- BUG FIX: Replaced imag() with anti_hermitian_part() ---
  // This correctly computes the anti-hermitian part of the plaquette matrix,
  // which is proportional to the field strength tensor F_munu.
  KOKKOS_FORCEINLINE_FUNCTION RealMatrix
  anti_hermitian_part(const SUN<Nc> &in) const {
    RealMatrix out{0};
#pragma unroll
    for (int i = 0; i < Nc; ++i) {
#pragma unroll
      for (int j = 0; j < Nc; ++j) {
        // Corresponds to (in - in^dagger) / (2i)
        out[i][j] = (in[i][j].imag() + Kokkos::conj(in[j][i]).imag()) / 2.0;
      }
    }
    return out;
  }

  KOKKOS_FORCEINLINE_FUNCTION real_t trace(const RealMatrix &in) const {
    real_t out{0};
#pragma unroll
    for (int i = 0; i < Nc; ++i) {
      out += in[i][i];
    }
    return out;
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION RealMatrix
  get_clover(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3, index_t mu, index_t nu) const {

    SUN<Nc> P_munu = zeroSUN<Nc>();
    const IndexArray<Nd> x{static_cast<index_t>(i0), static_cast<index_t>(i1),
                           static_cast<index_t>(i2), static_cast<index_t>(i3)};

    auto mod = [&](index_t val, index_t dir) {
      return (val + this->dimensions[dir]) % this->dimensions[dir];
    };

    // 1. Plaquette in (+mu, +nu) plane
    IndexArray<Nd> x_p_mu = x;
    x_p_mu[mu] = mod(x[mu] + 1, mu);
    IndexArray<Nd> x_p_nu = x;
    x_p_nu[nu] = mod(x[nu] + 1, nu);
    P_munu += g_in(x, mu) * g_in(x_p_mu, nu) * conj(g_in(x_p_nu, mu)) *
              conj(g_in(x, nu));

    // 2. Plaquette in (+mu, -nu) plane
    IndexArray<Nd> x_m_nu = x;
    x_m_nu[nu] = mod(x[nu] - 1, nu);
    IndexArray<Nd> x_p_mu_m_nu = x_p_mu;
    x_p_mu_m_nu[nu] = mod(x_p_mu[nu] - 1, nu);
    P_munu += g_in(x, mu) * conj(g_in(x_p_mu_m_nu, nu)) *
              conj(g_in(x_m_nu, mu)) * g_in(x_m_nu, nu);

    // 3. Plaquette in (-mu, +nu) plane
    IndexArray<Nd> x_m_mu = x;
    x_m_mu[mu] = mod(x[mu] - 1, mu);
    IndexArray<Nd> x_m_mu_p_nu = x_m_mu;
    x_m_mu_p_nu[nu] = mod(x_m_mu[nu] + 1, nu);
    P_munu += conj(g_in(x_m_mu, mu)) * g_in(x_m_mu, nu) *
              g_in(x_m_mu_p_nu, mu) * conj(g_in(x, nu));

    // 4. Plaquette in (-mu, -nu) plane
    IndexArray<Nd> x_m_mu_m_nu = x_m_mu;
    x_m_mu_m_nu[nu] = mod(x_m_mu[nu] - 1, nu);
    P_munu += conj(g_in(x_m_mu, mu)) * conj(g_in(x_m_mu_m_nu, nu)) *
              g_in(x_m_mu_m_nu, mu) * g_in(x_m_nu, nu);

    return anti_hermitian_part(P_munu); // Use the corrected function here
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

  TopoCharge<DGaugeFieldType> TCharge(g_in);
  tune_and_launch_for<Nd>("Calculate topological charge", IndexArray<Nd>{0},
                          g_in.dimensions, TCharge);
  Kokkos::fence();

  real_t charge_sum = TCharge.charge_per_site.sum();
  Kokkos::fence();

  return charge_sum / (32.0 * PI * PI);
}
} // namespace klft
