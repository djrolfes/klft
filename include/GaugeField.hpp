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
                     const complex_t init) {
      do_init(L0, L1, L2, L3, field, init);
    }

    template <class RNG>
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng, const real_t delta) {
      do_init(L0, L1, L2, L3, field, rng, delta);
    }

    template <class RNG>
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng) {
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

    // define accessors for the field
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const index_t i, const index_t j, const index_t k, const index_t l, const int mu) const {
      return field(i,j,k,l,mu);
    }
  
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const index_t i, const index_t j, const index_t k, const index_t l, const int mu) {
      return field(i,j,k,l,mu);
    }
    
  };

}
