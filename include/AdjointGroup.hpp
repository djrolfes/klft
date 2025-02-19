#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <typename T>
  struct SU3Generators {
    static constexpr T sqrt3 = T(1.7320508075688772935274463415059); //is this precise enough?
    // T1 = (1/2)*[ [0, 1, 0],
    //              [1, 0, 0],
    //              [0, 0, 0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T1 {{
        Kokkos::complex<T>(0, 0),      Kokkos::complex<T>(T(0.5), 0),  Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(T(0.5), 0),  Kokkos::complex<T>(0, 0),      Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),      Kokkos::complex<T>(0, 0),      Kokkos::complex<T>(0, 0)
    }};

    // T2 = (1/2)*[ [0, -i, 0],
    //              [i,  0, 0],
    //              [0,  0, 0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T2 {{
        Kokkos::complex<T>(0, 0),           Kokkos::complex<T>(0, -T(0.5)),  Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, T(0.5)),       Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),           Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0)
    }};

    // T3 = (1/2)*[ [1,  0, 0],
    //              [0, -1, 0],
    //              [0,  0, 0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T3 {{
        Kokkos::complex<T>(T(0.5), 0),   Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),        Kokkos::complex<T>(-T(0.5), 0),    Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),        Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0)
    }};

    // T4 = (1/2)*[ [0, 0, 1],
    //              [0, 0, 0],
    //              [1, 0, 0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T4 {{
        Kokkos::complex<T>(0, 0),      Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(T(0.5), 0),
        Kokkos::complex<T>(0, 0),      Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(T(0.5), 0),  Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0)
    }};

    // T5 = (1/2)*[ [0, 0, -i],
    //              [0, 0,  0],
    //              [i, 0,  0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T5 {{
        Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, -T(0.5)),
        Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, T(0.5)),     Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0)
    }};

    // T6 = (1/2)*[ [0, 0, 0],
    //              [0, 0, 1],
    //              [0, 1, 0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T6 {{
        Kokkos::complex<T>(0, 0),       Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),       Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(T(0.5), 0),
        Kokkos::complex<T>(0, 0),       Kokkos::complex<T>(T(0.5), 0),     Kokkos::complex<T>(0, 0)
    }};

    // T7 = (1/2)*[ [0, 0, 0],
    //              [0, 0, -i],
    //              [0, i,  0] ]
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T7 {{
        Kokkos::complex<T>(0, 0),       Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),       Kokkos::complex<T>(0, 0),         Kokkos::complex<T>(0, -T(0.5)),
        Kokkos::complex<T>(0, 0),       Kokkos::complex<T>(0, T(0.5)),     Kokkos::complex<T>(0, 0)
    }};

    // T8 = (1/2)*[ [1/√3,   0,       0],
    //              [0,     1/√3,     0],
    //              [0,      0,     -2/√3] ]
    // Note: Dividing by 2 gives: diag(1/(2√3), 1/(2√3), -1/√3)
    static constexpr Kokkos::Array<Kokkos::complex<T>, 9> T8 {{
        Kokkos::complex<T>(T(1.0/(2.0*sqrt3)), 0),  Kokkos::complex<T>(0, 0),                           Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),                              Kokkos::complex<T>(T(1.0/(2.0*sqrt3)), 0),  Kokkos::complex<T>(0, 0),
        Kokkos::complex<T>(0, 0),                              Kokkos::complex<T>(0, 0),                           Kokkos::complex<T>(- T(1.0/sqrt3), 0)
    }};

    static constexpr Kokkos::Array<Kokkos::Array<Kokkos::complex<T>, 9>, 8> getGenerators(){
      return {T1, T2, T3, T4, T5, T6, T7, T8};
    }
  };
  // Define a traits struct for the adjoint dimension.



  template <typename T>
  struct AdjointSU3 {
    static constexpr int nElements = 8;
      Kokkos::Array<T,8> v;

    AdjointSU3() = default;

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointSU3(const SU3<Tin> &in) {
      SU3<T> tmp((in - in.dagger())*0.5);
      Kokkos::complex<T> tmp_trace = 1/3 * tmp.trace();
      tmp.v[0] -= tmp_trace;
      tmp.v[4] -= tmp_trace;
      tmp.v[7] -= tmp_trace;
      Kokkos::Array<Kokkos::complex<T>, 9> SU3Gen;
      for (int i = 0; i < 8; i++){
        SU3Gen = SU3Generators<Tin>::getGenerators()[i];
        SU3<T> genTmp(SU3Gen); //abuse the SU3 class for matrix multiplication
        v[i] = 2*(genTmp*tmp).imtrace();
      }
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointSU3(const Tin &v1,const Tin &v2,const Tin &v3,const Tin &v4,const Tin &v5,const Tin &v6,const Tin &v7,const Tin &v8 ) {
      v[0] = v1;
      v[1] = v2;
      v[2] = v3;
      v[3] = v4;
      v[4] = v5;
      v[5] = v6;
      v[6] = v7;
      v[7] = v8;
    }

    KOKKOS_INLINE_FUNCTION AdjointSU3(Kokkos::Array<T,8> &in) {
      for(int i = 0; i<8; i++){
        v[i] = in[i];
      }
    }

    KOKKOS_INLINE_FUNCTION Kokkos::Array<Kokkos::complex<T>, 9> asMatrix() const {
      // once again abuse SU3 objects as Matrices
      SU3<T> tmp(Kokkos::complex<T>(0.0, 0.0));
      //tmp.set_identity();
      for (int i = 0; i < 8; i++){
        SU3<T> genTmp(SU3Generators<T>::getGenerators()[i]);
        tmp += genTmp*v[i];
      }
      return tmp.v;
    }

    KOKKOS_INLINE_FUNCTION T norm2() {
      return v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] + v[4]*v[4] + v[5]*v[5] + v[6]*v[6] + v[7]*v[7];
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator) {
      v[0] = generator.normal(0.,1.);
      v[1] = generator.normal(0.,1.);
      v[2] = generator.normal(0.,1.);
      v[3] = generator.normal(0.,1.);
      v[4] = generator.normal(0.,1.);
      v[5] = generator.normal(0.,1.);
      v[6] = generator.normal(0.,1.);
      v[7] = generator.normal(0.,1.);
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) const {
      return v[i];
    }

    KOKKOS_INLINE_FUNCTION void operator=(const AdjointSU3<T> &in) {
      v[0] = in.v[0];
      v[1] = in.v[1];
      v[2] = in.v[2];
      v[3] = in.v[3];
      v[4] = in.v[4];
      v[5] = in.v[5];
      v[6] = in.v[6];
      v[7] = in.v[7]; 
    }

    KOKKOS_INLINE_FUNCTION void operator+=(const AdjointSU3<T> &in) {
      v[0] += in.v[0];
      v[1] += in.v[1];
      v[2] += in.v[2];
      v[3] += in.v[3];
      v[4] += in.v[4];
      v[5] += in.v[5];
      v[6] += in.v[6];
      v[7] += in.v[7];
    }

    KOKKOS_INLINE_FUNCTION void operator-=(const AdjointSU3<T> &in) {
      v[0] -= in.v[0];
      v[1] -= in.v[1];
      v[2] -= in.v[2];
      v[3] -= in.v[3];
      v[4] -= in.v[4];
      v[5] -= in.v[5];
      v[6] -= in.v[6];
      v[7] -= in.v[7];
    }
    
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION AdjointSU3<T> operator*(const T &a, const AdjointSU3<T> &b) {
    return AdjointSU3<T>(a*b.v[0],a*b.v[1],a*b.v[2],a*b.v[3],a*b.v[4],a*b.v[5],a*b.v[6],a*b.v[7]);
  }

  template <typename T>
    KOKKOS_INLINE_FUNCTION SU3<T> exp(const AdjointSU3<T> &a) {
    // as a first quick and dirty implementation, use the first few Taylor approxs

    // Abuse SU3 objects as 3x3 matrices once again
    SU3<T> ident;
    ident.set_identity();
    const SU3<T> X(a.asMatrix());
    
    //Use first Taylor approx. (for now?)
    return SU3<T>(ident + X * Kokkos::complex<T>(0.0, 1.0));// + X * X * (-0.5) - X*X*X * (Kokkos::complex<T>(0.0, -1.0)/6.0));
  }


  template <typename T>
  struct AdjointSU2 {
    static constexpr int nElements = 3;
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

    KOKKOS_INLINE_FUNCTION T norm2() {
      return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
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
    static constexpr int nElements = 1;
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

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION AdjointU1(const Kokkos::Array<Tin,1> &v_in) {
      v = v_in[0];
    }

    KOKKOS_INLINE_FUNCTION void flip_sign() {
      v = -v;
    }

    KOKKOS_INLINE_FUNCTION T norm2() {
      return v*v;
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator) {
      v = generator.normal(0.,1.);
    }

    KOKKOS_INLINE_FUNCTION T operator()(const int &i) const {
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