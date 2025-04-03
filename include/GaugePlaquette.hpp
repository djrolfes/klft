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
#include "GaugeField.hpp"
#include "SUN.hpp"
#include "Tuner.hpp"

namespace klft
{
  
  // define a function to calculate the gauge plaquette
  // U_{mu nu} (x) = Tr[ U_mu(x) U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x) ]
  // for SU(N) gauge group
  // the return value is not normalized
  template <size_t Nd, size_t Nc>
  real_t GaugePlaquette(const deviceGaugeField<Nd,Nc> g_in) {
    complex_t plaq = 0.0;
    // get the start and end indices
    IndexArray<Nd> start;
    IndexArray<Nd> end;
    for (index_t i = 0; i < Nd; ++i) {
      start[i] = 0;
      end[i] = g_in.field.extent(i);
    }

    // store the field in a const gauge field
    const constGaugeField<Nd,Nc> g(g_in.field);

    // temporary field for storing results per site
    // direct reduction is slow
    // this field will be summed over in the end
    Field plaq_per_site(Kokkos::view_alloc(Kokkos::WithoutInitializing, "plaq_per_site"), end[0], end[1], end[2], end[3]);

    // tune and launch the kernel
    tune_and_launch_for<Nd>(start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        // temp SUN matrices to store products
        SUN<Nc> lmu, lnu;
        // reduction variable for all mu and nu
        complex_t tmunu(0.0, 0.0);

        #pragma unroll
        for(index_t mu = 0; mu < Nd; ++mu) {
          #pragma unroll
          for(index_t nu = 0; nu < Nd; ++nu) {
            if(nu > mu) {
              const index_t i0pmu = mu == 0 ? (i0 + 1) % end[0] : i0;
              const index_t i1pmu = mu == 1 ? (i1 + 1) % end[1] : i1;
              const index_t i2pmu = mu == 2 ? (i2 + 1) % end[2] : i2;
              const index_t i3pmu = mu == 3 ? (i3 + 1) % end[3] : i3;
              const index_t i0pnu = nu == 0 ? (i0 + 1) % end[0] : i0;
              const index_t i1pnu = nu == 1 ? (i1 + 1) % end[1] : i1;
              const index_t i2pnu = nu == 2 ? (i2 + 1) % end[2] : i2;
              const index_t i3pnu = nu == 3 ? (i3 + 1) % end[3] : i3;
              
              // form the 2 half plaquettes
              lmu = g(i0,i1,i2,i3,mu) * g(i0pmu,i1pnu,i2pnu,i3pnu,nu);
              lnu = g(i0,i1,i2,i3,nu) * g(i0pnu,i1pmu,i2pmu,i3pmu,mu);

              // multiply the 2 half plaquettes
              // take the trace
              #pragma unroll
              for(index_t c1 = 0; c1 < Nc; ++c1) {
                #pragma unroll
                for(index_t c2 = 0; c2 < Nc; ++c2) {
                  tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
                }
              }
            }
          }
        }

        // store the result in the temporary field
        plaq_per_site(i0,i1,i2,i3) = tmunu;
      });

    // sum over all sites
    Kokkos::parallel_reduce("sum_plaq", Policy<Nd>(start, end),
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3, complex_t &lsum) {
        lsum += plaq_per_site(i0,i1,i2,i3);
      }, Kokkos::Sum<complex_t>(plaq));
    Kokkos::fence();

    return Kokkos::real(plaq);
  }
}