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

// define SUN operations

#pragma once
#include "GLOBAL.hpp"

namespace klft {
template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void
print_SUN(const SUN<Nc> &a, const std::string &name = "SUN Matrix") {
  printf("%s:\n", name.c_str());
  for (size_t i = 0; i < Nc; i++) {
    for (size_t j = 0; j < Nc; j++) {
      double re = a[i][j].real();
      double im = a[i][j].imag();
      printf("[%zu,%zu] = (% .20f, % .20f i)\n", i, j, re, im);
    }
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const SUN<Nc> &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][0] * b[0][j];
#pragma unroll
      for (size_t k = 1; k < Nc; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void operator*=(SUN<Nc> &a, const SUN<Nc> &b) {
  SUN<Nc> c = a * b;
  a = c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator+(const SUN<Nc> &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void operator+=(SUN<Nc> &a, const SUN<Nc> &b) {
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      a[i][j] += b[i][j];
    }
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator-(const SUN<Nc> &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void operator-=(SUN<Nc> &a, const SUN<Nc> &b) {
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      a[i][j] -= b[i][j];
    }
  }
}

template <size_t Nc, typename Tin>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const SUN<Nc> &a, const Tin &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][j] * b;
    }
  }
  return c;
}

template <size_t Nc, typename Tin>
KOKKOS_FORCEINLINE_FUNCTION void operator*=(SUN<Nc> &a, const Tin &b) {
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      a[i][j] *= b;
    }
  }
}
template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const complex_t &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a * b[i][j];
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> conj(const SUN<Nc> &a) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = Kokkos::conj(a[j][i]);
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION complex_t trace(const SUN<Nc> &a) {
  complex_t c(0.0, 0.0);
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
    c += a[i][i];
  }
  return c;
}
KOKKOS_FORCEINLINE_FUNCTION SUN<1> traceLessAntiHermitian(const SUN<1> &a) {
  SUN<1> res;
  res[0][0] = complex_t(0, a[0][0].imag());
  return res;
}
// KOKKOS_FORCEINLINE_FUNCTION SUN<2> traceLessAntiHermitian(const SUN<2>& M) {
//   SUN<2> A;
//   auto v00 = M[0][0];
//   auto v11 = M[1][1];

//   auto tri = 1 / 2 * (v00.imag() + v11.imag());
//   auto v01 = M[0][1];
//   auto v10 = M[1][0];
//   auto x01 = v01 - conj(v10);
//   auto x10 = -conj(x01);
//   A[0][0] = (v00.imag() - tri) * complex_t(0, 1);
//   A[0][1] = 0.5 * x01;
//   A[1][0] = 0.5 * x10;
//   A[1][1] = (v11.imag() - tri) * complex_t(0, 1);

//   return A;
// }

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> traceLessAntiHermitian(const SUN<Nc> &a) {
  SUN<Nc> res;
  complex_t tra = 0.0;
  res = (a - conj(a)) * 0.5;
  tra = trace(res);
  tra /= static_cast<real_t>(Nc);

// #pragma unroll
//   for (size_t i = 0; i < Nc; ++i) {
// #pragma unroll
//     for (size_t j = 0; j < Nc; ++j) {
//       res[i][j] = 0.5 * (a[i][j] - conj(a[j][i]));
//     }
//   }

// #pragma unroll
//   for (size_t i = 0; i < Nc; ++i) {
//     trace += res[i][i];
//   }

//   complex_t tr_avg = trace / static_cast<real_t>(Nc);
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
    res[i][i] -= tra;
  }

  return res;
}

// random SUN matrix generator
// need to be defined for each Nc

// template <size_t N = Nc, typename std::enable_if<N == 1, int>::type = 0,
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<1> &r, RNG &generator,
                                         real_t delta) {
  // SUN<1> r;
  r[0][0] = Kokkos::exp(
      complex_t(0.0, generator.drand(-delta * Kokkos::numbers::pi_v<real_t>,
                                     delta * Kokkos::numbers::pi_v<real_t>)));
  // return r;
}

// template <size_t N = Nc, typename std::enable_if<N == 2, int>::type = 0,
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<2> &r, RNG &generator,
                                         real_t delta) {
  // SUN<2> r;
  real_t alpha =
      generator.drand(0.0, delta * 2 * Kokkos::numbers::pi_v<real_t>);
  real_t u = generator.drand(-1.0, 1.0);
  real_t theta = generator.drand(0.0, 2.0 * Kokkos::numbers::pi_v<real_t>);
  real_t salpha = Kokkos::sin(alpha);
  real_t n1 = Kokkos::sqrt(1.0 - u * u) * Kokkos::sin(theta);
  real_t n2 = Kokkos::sqrt(1.0 - u * u) * Kokkos::cos(theta);
  r[0][0] = complex_t(Kokkos::cos(alpha), u * salpha);
  r[0][1] = complex_t(n1 * salpha, n2 * salpha);
  r[1][0] = complex_t(-r[0][1].real(), r[0][1].imag());
  r[1][1] = complex_t(r[0][0].real(), -r[0][0].imag());
  // return r;
}

// template <size_t N = Nc, typename std::enable_if<N == 3, int>::type = 0,
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<3> &r, RNG &generator,
                                         real_t delta) {
  // SUN<3> r;
  real_t r1[6], r2[6], norm, fact;
  complex_t z1[3], z2[3], z3[3], z;
  while (1) {
    for (int i = 0; i < 6; i++) {
      r1[i] = generator.drand(0, delta);
    }
    norm = Kokkos::sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2] +
                        r1[3] * r1[3] + r1[4] * r1[4] + r1[5] * r1[5]);
    if (1.0 != (1.0 + norm))
      break;
  }
  fact = 1.0 / norm;
  z1[0] = fact * complex_t(r1[0], r1[1]);
  z1[1] = fact * complex_t(r1[2], r1[3]);
  z1[2] = fact * complex_t(r1[4], r1[5]);
  while (1) {
    while (1) {
      for (int i = 0; i < 6; i++) {
        r2[i] = generator.drand(0, delta);
      }
      norm = Kokkos::sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2] +
                          r2[3] * r2[3] + r2[4] * r2[4] + r2[5] * r2[5]);
      if (1.0 != (1.0 + norm))
        break;
    }
    fact = 1.0 / norm;
    z2[0] = fact * complex_t(r2[0], r2[1]);
    z2[1] = fact * complex_t(r2[2], r2[3]);
    z2[2] = fact * complex_t(r2[4], r2[5]);
    z = Kokkos::conj(z1[0]) * z2[0] + Kokkos::conj(z1[1]) * z2[1] +
        Kokkos::conj(z1[2]) * z2[2];
    z2[0] -= z * z1[0];
    z2[1] -= z * z1[1];
    z2[2] -= z * z1[2];
    norm =
        Kokkos::sqrt(z2[0].real() * z2[0].real() + z2[0].imag() * z2[0].imag() +
                     z2[1].real() * z2[1].real() + z2[1].imag() * z2[1].imag() +
                     z2[2].real() * z2[2].real() + z2[2].imag() * z2[2].imag());
    if (1.0 != (1.0 + norm))
      break;
  }
  fact = 1.0 / norm;
  z2[0] *= fact;
  z2[1] *= fact;
  z2[2] *= fact;
  z3[0] = Kokkos::conj((z1[1] * z2[2]) - (z1[2] * z2[1]));
  z3[1] = Kokkos::conj((z1[2] * z2[0]) - (z1[0] * z2[2]));
  z3[2] = Kokkos::conj((z1[0] * z2[1]) - (z1[1] * z2[0]));
  r[0][0] = z1[0];
  r[0][1] = z1[1];
  r[0][2] = z1[2];
  r[1][0] = z2[0];
  r[1][1] = z2[1];
  r[1][2] = z2[2];
  r[2][0] = z3[0];
  r[2][1] = z3[1];
  r[2][2] = z3[2];
  // return r;
}

// restore the gauge symmetry
// this also must be defined for each Nc

KOKKOS_FORCEINLINE_FUNCTION
SUN<1> restoreSUN(const SUN<1> &a) {
  SUN<1> c;
  c[0][0] = a[0][0] / Kokkos::sqrt(a[0][0].real() * a[0][0].real() +
                                   a[0][0].imag() * a[0][0].imag());
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
void restoreSUN(SUN<1> &a) {
  a[0][0] /= Kokkos::sqrt(a[0][0].real() * a[0][0].real() +
                          a[0][0].imag() * a[0][0].imag());
}

KOKKOS_FORCEINLINE_FUNCTION
SUN<2> restoreSUN(const SUN<2> &a) {
  SUN<2> c;
  real_t norm = Kokkos::sqrt(
      a[0][0].real() * a[0][0].real() + a[0][0].imag() * a[0][0].imag() +
      a[0][1].real() * a[0][1].real() + a[0][1].imag() * a[0][1].imag());
  if (norm < std::numeric_limits<real_t>::epsilon()) {
    // If norm is too small, return identity matrix
    c[0][0] = complex_t(1.0, 0.0);
    c[0][1] = complex_t(0.0, 0.0);
    c[1][0] = complex_t(0.0, 0.0);
    c[1][1] = complex_t(1.0, 0.0);
    return c;
  }
  c[0][0] = a[0][0] / norm;
  c[0][1] = a[0][1] / norm;
  c[1][0] = a[1][0] / norm;
  c[1][1] = a[1][1] / norm;
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
void restoreSUN(SUN<2> &a) {
  real_t norm = Kokkos::sqrt(
      a[0][0].real() * a[0][0].real() + a[0][0].imag() * a[0][0].imag() +
      a[0][1].real() * a[0][1].real() + a[0][1].imag() * a[0][1].imag());
  a[0][0] /= norm;
  a[0][1] /= norm;
  a[1][0] /= norm;
  a[1][1] /= norm;
}

KOKKOS_FORCEINLINE_FUNCTION
SUN<3> restoreSUN(const SUN<3> &a) {
  SUN<3> c;
  real_t norm0 = Kokkos::sqrt((Kokkos::conj(a[0][0]) * a[0][0] +
                               Kokkos::conj(a[0][1]) * a[0][1] +
                               Kokkos::conj(a[0][2]) * a[0][2])
                                  .real());
  real_t norm1 = Kokkos::sqrt((Kokkos::conj(a[1][0]) * a[1][0] +
                               Kokkos::conj(a[1][1]) * a[1][1] +
                               Kokkos::conj(a[1][2]) * a[1][2])
                                  .real());
  c[0][0] = a[0][0] / norm0;
  c[0][1] = a[0][1] / norm0;
  c[0][2] = a[0][2] / norm0;
  c[1][0] = a[1][0] / norm1;
  c[1][1] = a[1][1] / norm1;
  c[1][2] = a[1][2] / norm1;
  c[2][0] = Kokkos::conj((c[0][1] * c[1][2]) - (c[0][2] * c[1][1]));
  c[2][1] = Kokkos::conj((c[0][2] * c[1][0]) - (c[0][0] * c[1][2]));
  c[2][2] = Kokkos::conj((c[0][0] * c[1][1]) - (c[0][1] * c[1][0]));
  c[1][0] = Kokkos::conj((c[2][1] * c[0][2]) - (c[2][2] * c[0][1]));
  c[1][1] = Kokkos::conj((c[2][2] * c[0][0]) - (c[2][0] * c[0][2]));
  c[1][2] = Kokkos::conj((c[2][0] * c[0][1]) - (c[2][1] * c[0][0]));
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
void restoreSUN(SUN<3> &a) {
  real_t norm0 = Kokkos::sqrt((Kokkos::conj(a[0][0]) * a[0][0] +
                               Kokkos::conj(a[0][1]) * a[0][1] +
                               Kokkos::conj(a[0][2]) * a[0][2])
                                  .real());
  real_t norm1 = Kokkos::sqrt((Kokkos::conj(a[1][0]) * a[1][0] +
                               Kokkos::conj(a[1][1]) * a[1][1] +
                               Kokkos::conj(a[1][2]) * a[1][2])
                                  .real());
  a[0][0] /= norm0;
  a[0][1] /= norm0;
  a[0][2] /= norm0;
  a[1][0] /= norm1;
  a[1][1] /= norm1;
  a[1][2] /= norm1;
  a[2][0] = Kokkos::conj((a[0][1] * a[1][2]) - (a[0][2] * a[1][1]));
  a[2][1] = Kokkos::conj((a[0][2] * a[1][0]) - (a[0][0] * a[1][2]));
  a[2][2] = Kokkos::conj((a[0][0] * a[1][1]) - (a[0][1] * a[1][0]));
  a[1][0] = Kokkos::conj((a[2][1] * a[0][2]) - (a[2][2] * a[0][1]));
  a[1][1] = Kokkos::conj((a[2][2] * a[0][0]) - (a[2][0] * a[0][2]));
  a[1][2] = Kokkos::conj((a[2][0] * a[0][1]) - (a[2][1] * a[0][0]));
}

} // namespace klft
