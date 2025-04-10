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

// this file defines functions to calculate
// Wilson Loops for different gauge fields
// and different gauge groups

#pragma once
#include "GaugeField.hpp"

namespace klft
{

  // define a function to calculate Wilson Loop of length Lmu and Lnu
  // in the mu - nu plane
  // not normalized
  template <size_t Nd, size_t Nc>
  real_t WilsonLoop_mu_nu(const deviceGaugeField<Nd,Nc> g_in,
                          const index_t mu, const index_t nu,
                          const index_t Lmu, const index_t Lnu) {
    complex_t Wmunu = 0.0;

    // get the start and end indices
    const auto & dimensions = g_in.field.layout().dimension;
    IndexArray<Nd> start;
    IndexArray<Nd> end;
    for (index_t i = 0; i < Nd; ++i) {
      start[i] = 0;
      end[i] = dimensions[i];
    }

    // store the field in a const gauge field
    const constGaugeField<Nd,Nc> g(g_in.field);

    // define 2 temporary SUNFields for storing the 2 Wilson lines
    // of length Lmu and Lnu along mu and nu directions respectively
    SUNField<Nc> Lmu(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Lmu"), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    SUNField<Nc> Lnu(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Lnu"), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);

    // tune and launch the kernel
    tune_and_launch_for<Nd>("WilsonLoop_GaugeField_Lmu_Lnu", start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        // temp SUN matrices to store products
        SUN<Nc> lmu = g(i0,i1,i2,i3,mu);
        SUN<Nc> lnu = g(i0,i1,i2,i3,nu);
        // get the x + mu indices
        const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
        const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
        const index_t i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
        const index_t i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
        // get the x + nu indices
        const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
        const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
        const index_t i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;
        const index_t i3pnu = nu == 3 ? (i3 + 1) % dimensions[3] : i3;
        // build Lmu
        for(index_t i = 1; i < Lmu; ++i) {
          lmu *= g(i0pmu,i1pmu,i2pmu,i3pmu,mu);
          // get the next x + mu indices
          const index_t i0pmu = mu == 0 ? (i0pmu + 1) % dimensions[0] : i0pmu;
          const index_t i1pmu = mu == 1 ? (i1pmu + 1) % dimensions[1] : i1pmu;
          const index_t i2pmu = mu == 2 ? (i2pmu + 1) % dimensions[2] : i2pmu;
          const index_t i3pmu = mu == 3 ? (i3pmu + 1) % dimensions[3] : i3pmu;
        }
        // build Lnu
        for(index_t i = 1; i < Lnu; ++i) {
          lnu *= g(i0pnu,i1pnu,i2pnu,i3pnu,nu);
          // get the next x + nu indices
          const index_t i0pnu = nu == 0 ? (i0pnu + 1) % dimensions[0] : i0pnu;
          const index_t i1pnu = nu == 1 ? (i1pnu + 1) % dimensions[1] : i1pnu;
          const index_t i2pnu = nu == 2 ? (i2pnu + 1) % dimensions[2] : i2pnu;
          const index_t i3pnu = nu == 3 ? (i3pnu + 1) % dimensions[3] : i3pnu;
        }
        // store the lines
        Lmu(i0,i1,i2,i3) = lmu;
        Lnu(i0,i1,i2,i3) = lnu;
      });

    // define a temp field for storing the per site Wilson loop
    Field Wmunu_per_site(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Wmunu_per_site"), dimensions[0], dimensions[1], dimensions[2], dimensions[3]);

    // store the lines in const fields
    const constSUNField<Nc> Lmu_const(Lmu);
    const constSUNField<Nc> Lnu_const(Lnu);

    // tune and launch the kernel
    tune_and_launch_for<Nd>("WilsonLoop_GaugeField_Wmunu_per_site", start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        // get x + Lmu indices
        const index_t i0pLmu = mu == 0 ? (i0 + Lmu) % dimensions[0] : i0;
        const index_t i1pLmu = mu == 1 ? (i1 + Lmu) % dimensions[1] : i1;
        const index_t i2pLmu = mu == 2 ? (i2 + Lmu) % dimensions[2] : i2;
        const index_t i3pLmu = mu == 3 ? (i3 + Lmu) % dimensions[3] : i3;
        // get x + Lnu indices
        const index_t i0pLnu = nu == 0 ? (i0 + Lnu) % dimensions[0] : i0;
        const index_t i1pLnu = nu == 1 ? (i1 + Lnu) % dimensions[1] : i1;
        const index_t i2pLnu = nu == 2 ? (i2 + Lnu) % dimensions[2] : i2;
        const index_t i3pLnu = nu == 3 ? (i3 + Lnu) % dimensions[3] : i3;
        // temp SUN matrices to store products
        SUN<Nc> lmu = Lmu_const(i0,i1,i2,i3) * Lnu_const(i0pLmu,i1pLmu,i2pLmu,i3pLmu);
        SUN<Nc> lnu = Lnu_const(i0,i1,i2,i3) * Lmu_const(i0pLnu,i1pLnu,i2pLnu,i3pLnu);
        // construct loop, only need to take trace
        complex_t tmunu;
        #pragma unroll
        for(index_t c1 = 0; c1 < Nc; ++c1) {
          tmunu = lmu[c1][0] * Kokkos::conj(lnu[c1][0]);
          #pragma unroll
          for(index_t c2 = 1; c2 < Nc; ++c2) {
            tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
          }
        }
        // store the result in the temporary field
        Wmunu_per_site(i0,i1,i2,i3) = tmunu;
      });
    // sum over all sites
    Kokkos::parallel_reduce("WilsonLoop_GaugeField_sum_Wmunu", Policy<Nd>(start, end),
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3, complex_t &lsum) {
        lsum += Wmunu_per_site(i0,i1,i2,i3);
      }, Kokkos::Sum<complex_t>(Wmunu));
    // return the Wilson loop
    return Kokkos::real(Wmunu);
  }

  // define a function to calculate the Wilson Loop of length L
  // in the spatial directions and T in the temporal direction
  // not normalized
  template <size_t Nd, size_t Nc>
  real_t WilsonLoop_temporal(const deviceGaugeField<Nd,Nc> g_in,
                             const index_t L, const index_t T) {
    real_t Wt = 0.0;

    #pragma unroll
    for(index_t mu = 0; mu < Nd - 1; ++mu) {
      Wt += WilsonLoop_mu_nu(g_in, mu, Nd - 1, L, T);
    }
    Kokkos::fence();

    return Wt;
  }

}