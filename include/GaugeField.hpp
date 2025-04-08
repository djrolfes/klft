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

// define structs for initializing gauge fields

#pragma once
#include "GLOBAL.hpp"
#include "Tuner.hpp"
#include "SUN.hpp"

namespace klft
{

  template <size_t Nd, size_t Nc>
  struct deviceGaugeField {

    deviceGaugeField() = delete;

    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     const complex_t init) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, init);
    }

    template <class RNG>
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng, const real_t delta) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, rng, delta);
    }

    template <class RNG>
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, rng);
    }

    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0,i1,i2,i3,mu)[c1][c2] = init;
              }
            }
          }
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, RNG &rng, const real_t delta) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0,i1,i2,i3,mu), generator, delta);
          }
          rng.free_state(generator);
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, RNG &rng) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0,i1,i2,i3,mu)[c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                                   generator.drand(-1.0, 1.0));
              }
            }
          }
        });
      Kokkos::fence();
    }

    GaugeField<Nd,Nc> field;
    const IndexArray<4> dimensions;

    // define accessors for the field
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const index_t i0, const index_t i1, const index_t i2, const index_t i3, const index_t mu) const {
      return field(i0,i1,i2,i3,mu);
    }
  
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const index_t i0, const index_t i1, const index_t i2, const index_t i3, const index_t mu) {
      return field(i0,i1,i2,i3,mu);
    }

    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(const index_t i0, const index_t i1, const index_t i2, const index_t i3, const index_t mu) const {
      SUN<Nc> temp = zeroSUN<Nc>();
      // get the x + mu indices
      const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const index_t i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
      const index_t i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
      // positive directions
      #pragma unroll
      for(index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        // do nothing for mu = nu
        if (nu == mu) continue;
        // get the x + nu indices
        const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
        const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
        const index_t i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;
        const index_t i3pnu = nu == 3 ? (i3 + 1) % dimensions[3] : i3;
        // get the staple
        temp += field(i0pmu,i1pmu,i2pmu,i3pmu,nu) * conj(field(i0pnu,i1pnu,i2pnu,i3pnu,mu))
              * conj(field(i0,i1,i2,i3,nu));
      } // loop over nu
      // negative directions
      #pragma unroll
      for(index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        // do nothing for mu = nu
        if (nu == mu) continue;
        // get the x + mu - nu indices
        const index_t i0pmu_mnu = nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
        const index_t i1pmu_mnu = nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
        const index_t i2pmu_mnu = nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;
        const index_t i3pmu_mnu = nu == 3 ? (i3pmu - 1 + dimensions[3]) % dimensions[3] : i3pmu;
        // get the x - nu indices
        const index_t i0mnu = nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
        const index_t i1mnu = nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
        const index_t i2mnu = nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;
        const index_t i3mnu = nu == 3 ? (i3 - 1 + dimensions[3]) % dimensions[3] : i3;
        // get the staple
        temp += conj(field(i0pmu_mnu,i1pmu_mnu,i2pmu_mnu,i3pmu_mnu,nu)) 
              * conj(field(i0mnu,i1mnu,i2mnu,i3mnu,mu)) * field(i0mnu,i1mnu,i2mnu,i3mnu,nu);
      } // loop over nu
      return temp;
    }
    
  };

}
