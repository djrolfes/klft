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
#include "SUN.hpp"
#include "Tuner.hpp"
#include "IndexHelper.hpp"

namespace klft
{
  // first define the necessary functor
  template <size_t rank, size_t Nc, GaugeFieldKind kind = GaugeFieldKind::Standard>
  struct TopoCharge {
    // this kernel is defined for rank = Nd 
    constexpr static const size_t Nd = rank;
    static_assert(Nd == 4, "Topological charge is only defined for 4D gauge fields.");
    // define the gauge field type
    using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc, kind>::type;
    const GaugeFieldType g_in;
    // define the field type
    using FieldType = typename DeviceScalarFieldType<rank>::type;
    FieldType charge_per_site;

    using RealMatrix = Kokkos::Array<Kokkos::Array<real_t,Nc>,Nc>;

    // define the dimensions of the given Field
    const IndexArray<rank> dimensions;

    TopoCharge(GaugeFieldType &g_in): g_in(g_in) : dimensions(g_in.dimensions){
        charge_per_site = FieldType(dimensions, real_t(0.0));
    }
    
    
    /// Return the 4-D Levi–Civita symbol ε_{μνρσ} for indices 0..3.
    /// Zero if any indices repeat; +1 or –1 otherwise.
    // This should be evaluated at compile time.
    constexpr int epsilon4(int mu, int nu, int rho, int sigma) {
        // repeats -> 0
        if (mu == nu || mu == rho || mu == sigma ||
            nu == rho || nu == sigma ||
            rho == sigma) {
        return 0;
        }
        // pack into array
        constexpr int N = 4;
        int idx[N] = { mu, nu, rho, sigma };
        // count inversions
        int inv = 0;
        for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (idx[i] > idx[j]) ++inv;
        }
        }
        // parity of inversion count gives sign
        return (inv % 2 == 0 ? +1 : -1);
    }


    // now define the topological charge calculation for a single site (should this also be parallized over some directions?)
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION void operator()(const indexType i0, const indexType i1, const indexType i2, const indexType i3) const {
        
        int rho;
        int sigma;
        int mu = 0;
        RealMatrix C1, C2;
        // TODO 12.05.: implement this according to 1708.00696 and think about how to reuse C_munu 
        #pragma unroll
        for (int nu = mu+1; nu < Nd; ++nu){
            // determine the remaining indices and then swap indeces as well as C_munu with C_rhosigma to get the whole index set
            switch (nu)
            {
            case 1:
                rho = 2;
                sigma = 3;
                break;
            case 2:
                rho = 1;
                sigma = 4;
                break;
            case 3:
                rho = 1;
                sigma = 3;
                break;
            default:
                break;
            }

            // now go through the different mu, nu, sigma, rho combinations. This might be an application for thread teams for more parallelization.
            C1 = get_clover(i0,i1,i2,i3, mu, nu);
            C2 = get_clover(i0,i1,i2,i3, rho, sigma);
            charge_per_site(i0, i1, i2, i3) += epsilon4(mu, nu, rho, sigma)*trace(C1*C2);
            charge_per_site(i0, i1, i2, i3) += epsilon4(rho, sigma, mu, nu)*trace(C2*C1);

            C1 = get_clover(i0,i1,i2,i3, mu, nu);
            C2 = get_clover(i0,i1,i2,i3, sigma, rho);
            charge_per_site(i0, i1, i2, i3) += epsilon4(mu, nu, sigma, rho)*trace(C1*C2);
            charge_per_site(i0, i1, i2, i3) += epsilon4(sigma, rho, mu, nu)*trace(C2*C1);

            C1 = get_clover(i0,i1,i2,i3, nu, mu);
            C2 = get_clover(i0,i1,i2,i3, rho, sigma);
            charge_per_site(i0, i1, i2, i3) += epsilon4(nu, mu, rho, sigma)*trace(C1*C2);
            charge_per_site(i0, i1, i2, i3) += epsilon4(rho, sigma, nu, mu)*trace(C2*C1);
            
            C1 = get_clover(i0,i1,i2,i3, nu, mu);
            C2 = get_clover(i0,i1,i2,i3, sigma, rho);
            charge_per_site(i0, i1, i2, i3) += epsilon4(nu, mu, sigma, rho)*trace(C1*C2);
            charge_per_site(i0, i1, i2, i3) += epsilon4(sigma, rho, nu, mu)*trace(C2*C1);
        }
        charge_per_site(i0, i1, i2, i3) /= 16;
    }

    // get the imaginary parts of an SUN matrix
    KOKKOS_FORCEINLINE_FUNCTION RealMatrix imag(const SUN<Nc> &in) const {
        RealMatrix out{0};
        #pragma unroll
        for (int i = 0; i < Nc; ++i){
            #pragma unroll
            for (int j = 0; j < Nc; ++j){
                out[i][j] = in[i][j].imag();
            }
        }
    }

    // return the trace of a RealMatrix
    KOKKOS_FORCEINLINE_FUNCTION real_t trace(const RealMatrix &in) const {
        real_t out{0};
        #pragma unroll
        for(int i = 0; i < Nc; ++i){
            out += in[i][i];
        }
        return out;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    RealMatrix operator*(const RealMatrix &a, const RealMatrix &b) {
      RealMatrix c;
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



    // return the clover C_munu
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION RealMatrix get_clover(const indexType i0, const indexType i1, const indexType i2, const indexType i3, int mu, int nu) const {
        
        IndexArray site1{i0,i1,i2,i3};
        IndexArray site2{i0,i1,i2,i3};
        IndexArray site3{i0,i1,i2,i3};

        site2[mu] = site2[mu] + 1 % this->dimensions[mu];
        site3[nu] = site3[nu] + 1 % this->dimensions[nu];
        SUN<Nc> P = this->g_in(site1,mu)* this->g_in(site2,nu).conj() * this->g_in(site3,mu).conj() * this->g_in(site1,nu);
        
        site2[mu] = site2[mu] - 2 + this->dimensions[mu] % this->dimensions[mu]; // -mu
        site3[mu] = site3[mu] - 1 + this->dimensions[mu] % this->dimensions[mu]; // +nu,-mu
        P += this->g_in(site1, nu) * this->g_in(site3, mu).conj() * this->g_in(site2, nu).conj() * this->g_in(site2, mu);

        site2[mu] = site2[mu] + 1 % this->dimensions[mu]; // -0
        site2[nu] = site2[nu] - 1 + this->dimensions[nu]% this->dimensions[nu]; // -nu
        site3[mu] = site3[mu] + 2 % this->dimensions[mu]; // +nu,+mu
        site3[nu] = site3[nu] - 2 + this->dimensions[nu] % this->dimensions[nu]; // -nu,+mu
        P += this->g_in(site2, nu).conj() * this->g_in(site3, nu).conj() * this->g_in(site3, mu) * this->g_in(site1, mu).conj();

        site3[mu] = site3[mu] - 2 + this->dimensions[mu] % this->dimensions[mu]; // -nu,-mu
        site1[mu] = site1[mu] - 1 + this->dimensions[mu] % this->dimensions[mu]; // -mu
        P += this->g_in(site1, mu).conj() * this->g_in(site3, nu).conj() * this->g_in(site3, mu) * this->g_in(site2, nu);
        return imag(P);
    }

  };


  template <size_t rank, size_t Nc, GaugeFieldKind kind = GaugeFieldKind::Standard>
  real_t get_topological_charge(const typename DeviceGaugeFieldType<rank, Nc, kind>::type &g_in){
    constexpr static const size_t Nd = rank;

    // define the functor
    TopoCharge<rank, Nc, kind> TCharge(g_in);
    tune_and_launch<rank>("Calculate topological charge", IndexArray<rank>{0}, g_in.dimensions, TCharge);
    Kokkos::fence();

    real_t charge = TCharge.charge_per_site.sum();
    Kokkos::fence();
    charge /= 32 * Kokkos::number::pi<real_t> * Kokkos::number::pi<real_t>;

    return charge;

  }
} //namespace klft
