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

// define sun (Lie algebra of SUN) operations

#pragma once
#include "GLOBAL.hpp"
#include "SUN.hpp"

namespace klft {

    // conversion Functionst going from SUN to sun
    // su2 and u1 use the same representation used in the earlier AdjointGroup implementation
    KOKKOS_FORCEINLINE_FUNCTION sun<2> from_Group(const SUN<2> &g){
        complex_t a{g[0][0]}, b{g[0][1]};
        sun<2> rtn {2.0*b.imag(), 2.0*b.real(), 2.0*a.imag()};
        return rtn;
    }

    KOKKOS_FORCEINLINE_FUNCTION sun<1> from_Group(const SUN<1> &g){
        complex_t a{g[0][0]};
        sun<1> rtn {a.imag()};
        return rtn;
    }


    // TODO: implement for su3 using the linear parameters of the group generators


    // exponentiate the su2 element to return the corresponding SU2 group element
    KOKKOS_FORCEINLINE_FUNCTION SUN<2> exp(const sun<2> &in){
        const real_t alpha = Kokkos::sqrt(in[0]*in[0] + in[1]*in[1] + in[2]*in[2]);
        Kokkos::Array<real_t,3> n = {in[0]/alpha,in[1]/alpha,in[2]/alpha};
        const real_t salpha = Kokkos::sin(alpha);
        SUN<2> rtn;
        rtn[0][0] = complex_t(Kokkos::cos(alpha),n[2]*salpha);
        rtn[0][1] = complex_t(n[1]*salpha,n[0]*salpha);
        rtn[1][0] = -rtn[0][1];
        rtn[1][1] = complex_t(Kokkos::cos(alpha),-n[2]*salpha);
        return rtn;
    }

    // exponentiate the u1 element to return the corresponding U1 group element
    KOKKOS_FORCEINLINE_FUNCTION SUN<1> exp(const sun<1> &in){
        SUN<1> rtn;
        rtn [0][0] = complex_t(Kokkos::cos(in[0]), Kokkos::sin(in[0]));
        return rtn;
    }

    // TODO: implement exp for an su3 element (see quda for inspiration)

    //define operators for sun<> objects
    template <size_t Nc>
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator*=(sun<Nc> &a, const real_t &scalar){
        #pragma unroll
        for (int i=0; i<Kokkos::max<size_t>(Nc * Nc-1, 1); ++i){
            a[i] *= scalar;
        }
        return a;
    }

    template <size_t Nc>
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator*(const sun<Nc> &a, const real_t &scalar){
        sun<Nc> ret{};
        #pragma unroll
        for (int i=0; i<Kokkos::max<size_t>(Nc * Nc-1, 1); ++i){
            ret[i] = a[i] * scalar;
        }
        return ret;
    }

    template <size_t Nc>
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator*(const real_t &scalar, const sun<Nc> &a){
        return a * scalar;
    }

    template <size_t Nc> 
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator+(const sun<Nc> &a, const sun<Nc> &b){
        sun<Nc> ret{};
        #pragma unroll
        for (int i=0; i<Kokkos::max<size_t>(Nc * Nc-1, 1); ++i){
            ret[i] = a[i] + b[i];
        }
        return ret;
    }

    template <size_t Nc> 
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator+=(sun<Nc> &a, const sun<Nc> &b){
        #pragma unroll
        for (int i=0; i<Kokkos::max<size_t>(Nc * Nc-1, 1); ++i){
            a[i] += b[i];
        }
        return a;
    }

    template <size_t Nc> 
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator-(const sun<Nc> &a, const sun<Nc> &b){
        sun<Nc> ret{};
        #pragma unroll
        for (int i=0; i<Kokkos::max<size_t>(Nc * Nc-1, 1); ++i){
            ret[i] = a[i] - b[i];
        }
        return ret;
    }

    template <size_t Nc> 
    KOKKOS_FORCEINLINE_FUNCTION sun<Nc> operator-=(sun<Nc> &a, const sun<Nc> &b){
        #pragma unroll
        for (int i=0; i<Kokkos::max<size_t>(Nc * Nc-1, 1); ++i){
            a[i] -= b[i];
        }
        return a;
    }


}