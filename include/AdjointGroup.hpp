#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <typename T>
  struct AdjointSU2 {

    Kokkos::Array<T,3> v;

    AdjointSU2() = default;

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointSU2(const SU2<Tin> &in) {
      Kokkos::complex a = in.v[0], b = in.v[1];
      v[0] = 2.0*b.imag();
      v[1] = 2.0*b.real();
      v[2] = 2.0*a.imag();
    }

    KOKKOS_INLINE_FUNCTION AdjointSU2(const T &a, const T &b, const T &c) {
      v[0] = a;
      v[1] = b;
      v[2] = c;
    }

    KOKKOS_INLINE_FUNCTION AdjointSU2(const Kokkos::Array<T,3> &v_in) {
      v[0] = v_in[0];
      v[1] = v_in[1];
      v[2] = v_in[2];
    }

    KOKKOS_INLINE_FUNCTION AdjointSU2(const T v_in[3]) {
      v[0] = v_in[0];
      v[1] = v_in[1];
      v[2] = v_in[2];
    }

    KOKKOS_INLINE_FUNCTION AdjointSU2(const T &v_in) {
      v[0] = v_in;
      v[1] = v_in;
      v[2] = v_in;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointSU2(const AdjointSU2<Tin> &in) {
      v[0] = in.v[0];
      v[1] = in.v[1];
      v[2] = in.v[2];
    }

    KOKKOS_INLINE_FUNCTION void flip_sign() {
      v[0] = -v[0];
      v[1] = -v[1];
      v[2] = -v[2];
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator) {
      v[0] = generator.normal(0.,1.);
      v[1] = generator.normal(0.,1.);
      v[2] = generator.normal(0.,1.);
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) const {
      return v[i];
    }

    KOKKOS_INLINE_FUNCTION void operator=(const AdjointSU2<T> &in) {
      v[0] = in.v[0];
      v[1] = in.v[1];
      v[2] = in.v[2];
    }

    KOKKOS_INLINE_FUNCTION void operator+=(const AdjointSU2<T> &in) {
      v[0] += in.v[0];
      v[1] += in.v[1];
      v[2] += in.v[2];
    }

    KOKKOS_INLINE_FUNCTION void operator-=(const AdjointSU2<T> &in) {
      v[0] -= in.v[0];
      v[1] -= in.v[1];
      v[2] -= in.v[2];
    }

  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION AdjointSU2<T> operator*(const T &a, const AdjointSU2<T> &b) {
    return AdjointSU2<T>(a*b.v[0],a*b.v[1],a*b.v[2]);
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION SU2<T> exp(const AdjointSU2<T> &a) {
    const T alpha = Kokkos::sqrt(a.v[0]*a.v[0] + a.v[1]*a.v[1] + a.v[2]*a.v[2]);
    Kokkos::Array<T,3> n = {a.v[0]/alpha,a.v[1]/alpha,a.v[2]/alpha};
    const T salpha = Kokkos::sin(alpha);
    return SU2<T>(Kokkos::complex(Kokkos::cos(alpha),n[2]*salpha),Kokkos::complex(n[1]*salpha,n[0]*salpha),
                  Kokkos::complex(-n[1]*salpha,n[0]*salpha),Kokkos::complex(Kokkos::cos(alpha),-n[2]*salpha));
  }

  template <typename T>
  struct AdjointU1 {

    T v;

    AdjointU1() = default;

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointU1(const U1<Tin> &in) {
      v = in.v.imag();
    }

    KOKKOS_INLINE_FUNCTION AdjointU1(const T &v_in) {
      v = v_in;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointU1(const AdjointU1<Tin> &in) {
      v = in.v;
    }

    KOKKOS_INLINE_FUNCTION void flip_sign() {
      v = -v;
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator) {
      v = generator.normal(0.,1.);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) const {
      return v;
    }

    KOKKOS_INLINE_FUNCTION void operator=(const AdjointU1<T> &in) {
      v = in.v;
    }

    KOKKOS_INLINE_FUNCTION void operator+=(const AdjointU1<T> &in) {
      v += in.v;
    }

    KOKKOS_INLINE_FUNCTION void operator-=(const AdjointU1<T> &in) {
      v -= in.v;

    }

  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION AdjointU1<T> operator*(const T &a, const AdjointU1<T> &b) {
    return AdjointU1<T>(a*b.v);
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION U1<T> exp(const AdjointU1<T> &a) {
    return U1<T>(Kokkos::complex(Kokkos::cos(a.v),Kokkos::sin(a.v)));
  }

} // namespace klft