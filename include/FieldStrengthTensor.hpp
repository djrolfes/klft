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
#include <unistd.h>

namespace klft {

// TODO: move these operators somewhere more appropriate
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

template <typename T, typename U, size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<T, N>, N>
operator*(const Kokkos::Array<Kokkos::Array<T, N>, N> &a, const U &b) {
  Kokkos::Array<Kokkos::Array<T, N>, N> c;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      c[i][j] = a[i][j] * b;
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

// get the imaginary parts of an SUN matrix
template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<real_t, N>, N>
imag(const SUN<N> &in) {
  Kokkos::Array<Kokkos::Array<real_t, N>, N> out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      out[i][j] = in[i][j].imag();
    }
  }
  return out;
}

// return the trace of a RealMatrix
template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION real_t
trace(const Kokkos::Array<Kokkos::Array<real_t, N>, N> &in) {
  real_t out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
    out += in[i][i];
  }
  return out;
}

template <typename DGaugeFieldType> struct FieldStrengthTensor {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(
      Nd == 4,
      "FieldStrengthTensor is only defined for 4D gauge fields. (for now)");
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  // define the gauge field type
  using GaugeFieldType = typename DGaugeFieldType::type;
  const GaugeFieldType g_in;
  // define the field type
  using FieldType = typename DeviceScalarFieldType<Nd>::type;

  struct CloverDef {};

  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;

  // define the dimensions of the given Field
  const IndexArray<Nd> dimensions;

  FieldStrengthTensor(const GaugeFieldType g_in)
      : g_in(g_in), dimensions(g_in.dimensions) {}

  // return the clover C_munu
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  operator()(CloverDef, const indexType i0, const indexType i1,
             const indexType i2, const indexType i3, index_t mu,
             index_t nu) const {

    SUN<Nc> P_munu = zeroSUN<Nc>();
    const IndexArray<Nd> x{static_cast<index_t>(i0), static_cast<index_t>(i1),
                           static_cast<index_t>(i2), static_cast<index_t>(i3)};

    // Helper lambda for modular arithmetic
    auto mod = [&](index_t s, index_t dir) {
      return (s + this->dimensions[dir]);
    };

    // 1. Plaquette in (+mu, +nu) plane starting at x
    IndexArray<Nd> x_p_mu = x;
    x_p_mu[mu] = (x[mu] + 1) % dimensions[mu];
    IndexArray<Nd> x_p_nu = x;
    x_p_nu[nu] = (x[nu] + 1) % dimensions[nu];
    P_munu += g_in(x, mu) * g_in(x_p_mu, nu) * conj(g_in(x_p_nu, mu)) *
              conj(g_in(x, nu));

    // 2. Plaquette in (+mu, -nu) plane starting at x
    IndexArray<Nd> x_m_nu = x;
    x_m_nu[nu] = mod(x[nu] - 1, nu) % dimensions[nu];
    IndexArray<Nd> x_p_mu_m_nu = x_p_mu;
    x_p_mu_m_nu[nu] = mod(x_p_mu[nu] - 1, nu) % dimensions[nu];
    P_munu += g_in(x, mu) * conj(g_in(x_p_mu_m_nu, nu)) *
              conj(g_in(x_m_nu, mu)) * g_in(x_m_nu, nu);

    // 3. Plaquette in (-mu, +nu) plane starting at x
    IndexArray<Nd> x_m_mu = x;
    x_m_mu[mu] = mod(x[mu] - 1, mu) % dimensions[mu];
    IndexArray<Nd> x_m_mu_p_nu = x_m_mu;
    x_m_mu_p_nu[nu] = (x_m_mu[nu] + 1) % dimensions[nu];
    P_munu += conj(g_in(x_m_mu, mu)) * g_in(x_m_mu, nu) *
              g_in(x_m_mu_p_nu, mu) * conj(g_in(x, nu));

    // 4. Plaquette in (-mu, -nu) plane starting at x
    IndexArray<Nd> x_m_mu_m_nu = x_m_mu;
    x_m_mu_m_nu[nu] = mod(x_m_mu[nu] - 1, nu) % dimensions[nu];
    P_munu += conj(g_in(x_m_mu, mu)) * conj(g_in(x_m_mu_m_nu, nu)) *
              g_in(x_m_mu_m_nu, mu) * g_in(x_m_nu, nu);

    return imag(P_munu) * 0.25;
  }
};

} // namespace klft
