#pragma once
#include "GLOBAL.hpp"

namespace klft {

  template <typename T>
  struct SU3 {

    Kokkos::Array<T,9> v;

    SU3() = default;

    KOKKOS_INLINE_FUNCTION SU3(const T &a, const T &b, const T &c, const T &d, const T &e, const T &f, const T &g, const T &h, const T &i) {
      this->v[0] = a;
      this->v[1] = b;
      this->v[2] = c;
      this->v[3] = d;
      this->v[4] = e;
      this->v[5] = f;
      this->v[6] = g;
      this->v[7] = h;
      this->v[8] = i;
    }

    KOKKOS_INLINE_FUNCTION SU3(const T v_in[9]) {
      this->v[0] = v_in[0];
      this->v[1] = v_in[1];
      this->v[2] = v_in[2];
      this->v[3] = v_in[3];
      this->v[4] = v_in[4];
      this->v[5] = v_in[5];
      this->v[6] = v_in[6];
      this->v[7] = v_in[7];
      this->v[8] = v_in[8];
    }

    KOKKOS_INLINE_FUNCTION SU3(const T &v_in) {
      this->v[0] = v_in;
      this->v[1] = v_in;
      this->v[2] = v_in;
      this->v[3] = v_in;
      this->v[4] = v_in;
      this->v[5] = v_in;
      this->v[6] = v_in;
      this->v[7] = v_in;
      this->v[8] = v_in;
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      this->v[0] = 1.0;
      this->v[1] = 0.0;
      this->v[2] = 0.0;
      this->v[3] = 0.0;
      this->v[4] = 1.0;
      this->v[5] = 0.0;
      this->v[6] = 0.0;
      this->v[7] = 0.0;
      this->v[8] = 1.0;
    } 
  };

  template <typename T>
  struct SU2 {

    Kokkos::Array<T,4> v;

    SU2() = default;

    KOKKOS_INLINE_FUNCTION SU2(const T &a, const T &b, const T &c, const T &d) {
      this->v[0] = a;
      this->v[1] = b;
      this->v[2] = c;
      this->v[3] = d;
    }

    KOKKOS_INLINE_FUNCTION SU2(const T v_in[4]) {
      this->v[0] = v_in[0];
      this->v[1] = v_in[1];
      this->v[2] = v_in[2];
      this->v[3] = v_in[3];
    }

    KOKKOS_INLINE_FUNCTION SU2(const T &v_in) {
      this->v[0] = v_in;
      this->v[1] = v_in;
      this->v[2] = v_in;
      this->v[3] = v_in;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2(const SU2<Tin> &in) {
      this->v[0] = in.v[0];
      this->v[1] = in.v[1];
      this->v[2] = in.v[2];
      this->v[3] = in.v[3];
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      this->v[0] = 1.0;
      this->v[1] = 0.0;
      this->v[2] = 0.0;
      this->v[3] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) {
      return this->v[i];
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) const {
      return this->v[i];
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      this->v[1] = -this->v[1];
      this->v[2] = -this->v[2];
      this->v[3] = -this->v[3];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const SU2<Tin> &in) {
      this->v[0] += in.v[0];
      this->v[1] += in.v[1];
      this->v[2] += in.v[2];
      this->v[3] += in.v[3];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const SU2<Tin> &in) {
      this->v[0] -= in.v[0];
      this->v[1] -= in.v[1];
      this->v[2] -= in.v[2];
      this->v[3] -= in.v[3];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator+(const SU2<Tin> &in) const {
      SU2<T> tmp(this->v[0] + in.v[0], this->v[1] + in.v[1], this->v[2] + in.v[2], this->v[3] + in.v[3]);
      return tmp;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator-(const SU2<Tin> &in) const {
      SU2<T> tmp(this->v[0] - in.v[0], this->v[1] - in.v[1], this->v[2] - in.v[2], this->v[3] - in.v[3]);
      return tmp;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const SU2<Tin> &in) {
      T a = this->v[0]*in.v[0] - this->v[1]*in.v[1] - this->v[2]*in.v[2] - this->v[3]*in.v[3];
      T b = this->v[0]*in.v[1] + this->v[1]*in.v[0] + this->v[2]*in.v[3] - this->v[3]*in.v[2];
      T c = this->v[0]*in.v[2] - this->v[1]*in.v[3] + this->v[2]*in.v[0] + this->v[3]*in.v[1];
      T d = this->v[0]*in.v[3] + this->v[1]*in.v[2] - this->v[2]*in.v[1] + this->v[3]*in.v[0];
      this->v[0] = a;
      this->v[1] = b;
      this->v[2] = c;
      this->v[3] = d;
    }
  
    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator*(const SU2<Tin> &in) const {
      T a = this->v[0]*in.v[0] - this->v[1]*in.v[1] - this->v[2]*in.v[2] - this->v[3]*in.v[3];
      T b = this->v[0]*in.v[1] + this->v[1]*in.v[0] + this->v[2]*in.v[3] - this->v[3]*in.v[2];
      T c = this->v[0]*in.v[2] - this->v[1]*in.v[3] + this->v[2]*in.v[0] + this->v[3]*in.v[1];
      T d = this->v[0]*in.v[3] + this->v[1]*in.v[2] - this->v[2]*in.v[1] + this->v[3]*in.v[0];
      SU2<T> tmp(a, b, c, d);
      return tmp;
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return this->v[0]*2.0;
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      T norm = Kokkos::sqrt(this->v[0]*this->v[0] + this->v[1]*this->v[1] + this->v[2]*this->v[2] + this->v[3]*this->v[3]);
      this->v[0] /= norm;
      this->v[1] /= norm;
      this->v[2] /= norm;
      this->v[3] /= norm;
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      T alpha = generator.drand(0.0,delta*2*Kokkos::numbers::pi_v<T>);
      T u = generator.drand(-1.0,1.0);
      T theta = generator.drand(0.0,2.0*Kokkos::numbers::pi_v<T>);
      T salpha = Kokkos::sin(alpha);
      T n1 = Kokkos::sqrt(1.0 - u*u)*Kokkos::sin(theta);
      T n2 = Kokkos::sqrt(1.0 - u*u)*Kokkos::cos(theta);
      this->v[0] = Kokkos::cos(alpha);
      this->v[1] = u*salpha;
      this->v[2] = n1*salpha;
      this->v[3] = n2*salpha;
    }
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION SU2<T> dagger(const SU2<T> &in) {
    SU2<T> tmp(in);
    tmp.dagger();
    return tmp;
  }

  // template <typename T, class GaugeGroup, class RNG>
  // KOKKOS_INLINE_FUNCTION GaugeGroup get_random(RNG &generator, T delta) {}

  // template <typename T, class RNG>
  // KOKKOS_INLINE_FUNCTION SU2<T> get_random(RNG &generator, T delta) {
  //   T alpha = generator.drand(0.0,delta*2*Kokkos::numbers::pi_v<T>);
  //   T u = generator.drand(-1.0,1.0);
  //   T theta = generator.drand(0.0,2.0*Kokkos::numbers::pi_v<T>);
  //   T salpha = Kokkos::sin(alpha);
  //   T n1 = Kokkos::sqrt(1.0 - u*u)*Kokkos::sin(theta);
  //   T n2 = Kokkos::sqrt(1.0 - u*u)*Kokkos::cos(theta);
  //   SU2<T> R(Kokkos::cos(alpha),u*salpha,n1*salpha,n2*salpha);
  //   return R;
  // }

  template <typename T>
  struct U1 {

    T v;

    U1() = default;

    KOKKOS_INLINE_FUNCTION U1(const T &a) {
      this->v = a;
    }

    KOKKOS_INLINE_FUNCTION U1(const U1<T> &in) {
      this->v = in.v;
    }

    KOKKOS_INLINE_FUNCTION U1(const T v_in[1]) {
      this->v = v_in[0];
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      this->v = 0.0;
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) {
      return this->v;
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) const {
      return this->v;
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      this->v = -this->v;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const U1<Tin> &in) {
      this->v = Kokkos::acos((Kokkos::exp(Kokkos::complex<T>(0.0,this->v)) + Kokkos::exp(Kokkos::complex<T>(0.0,in.v))).real());
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const U1<Tin> &in) {
      this->v = Kokkos::acos((Kokkos::exp(Kokkos::complex<T>(0.0,this->v)) - Kokkos::exp(Kokkos::complex<T>(0.0,in.v))).real());
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator+(const U1<Tin> &in) const {
      return U1<T>(Kokkos::acos((Kokkos::exp(Kokkos::complex<T>(0.0,this->v)) + Kokkos::exp(Kokkos::complex<T>(0.0,in.v))).real()));
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator-(const U1<Tin> &in) const {
      return U1<T>(Kokkos::acos((Kokkos::exp(Kokkos::complex<T>(0.0,this->v)) - Kokkos::exp(Kokkos::complex<T>(0.0,in.v))).real()));
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const U1<Tin> &in) {
      this->v += in.v;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator*(const U1<Tin> &in) const {
      return U1<T>(this->v + in.v);
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return Kokkos::cos(this->v);
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      this->v = Kokkos::acos(Kokkos::cos(this->v));
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      this->v = generator.drand(-delta*Kokkos::numbers::pi_v<T>,delta*Kokkos::numbers::pi_v<T>);
    }
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION U1<T> dagger(const U1<T> &in) {
    return U1<T>(-in.v);
  }

  // template <typename T, class G, class RNG, std::enable_if_t<std::is_same<G,U1<T>>::value,int> = 0>
  // KOKKOS_INLINE_FUNCTION U1<T> get_random(RNG &generator, T delta) {
  //   return U1<T>(generator.drand(-delta*Kokkos::numbers::pi_v<T>,delta*Kokkos::numbers::pi_v<T>));
  // }

} // namespace klqcd
